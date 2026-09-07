"""
03_bayesian_hierarchical.py  (v2)
=========================================================
베이지안 계층 회귀 모델 (Hierarchical / Partial Pooling Regression)

v1 대비 변경점 (교수님 코멘트 반영)
-----------------------------------
[C0] 사전분포(prior) 명시
     - 아래 "PRIOR SPECIFICATION" 섹션에 모든 파라미터의 사전분포와 선택 근거를 기술.
     - pm.sample_prior_predictive() 로 사전예측검정(prior predictive check)을 수행하고
       그림으로 저장하여, 사전분포가 물리적으로 말이 되는 범위를 덮는지 확인.

[C1] 그룹 효과를 random -> fixed 로 변경
     - v1: mu_group ~ Normal(mu_global, tau_group), 즉 그룹 절편을 랜덤효과로 두었음.
       그룹이 3개(urban / suburban_industrial / background)뿐이라 tau_group(그룹 간 분산)은
       사실상 데이터가 아니라 사전분포가 결정하게 되어 추정이 무의미하다.
     - v2: gamma_group ~ Normal(4.0, 1.0) 을 그룹마다 독립적으로 부여(= 고정효과).
       station 절편만 그룹 평균 주위로 부분 풀링한다.
         alpha_station[i] ~ Normal(gamma_group[g(i)], tau_station)

[C2] 잔차의 AR(1) 구조 추가
     - PM2.5는 자기상관이 강한 시계열이므로 관측치를 독립으로 두면 신용구간이 과소추정된다.
     - station별 일별 시계열에 대해 조건부 우도(conditional likelihood) 형태로 AR(1) 오차를 도입:
         y[i,t] ~ Normal( mu[i,t] + rho * (y[i,t-1] - mu[i,t-1]), sigma_e )
       각 station 시계열의 첫 관측(또는 날짜가 끊긴 직후 관측)은
         y[i,t] ~ Normal( mu[i,t], sigma_e / sqrt(1 - rho^2) )   (정상분포 주변분산)
       -> 잠재변수를 추가하지 않으므로 계산비용이 거의 늘지 않는다.

[C3] 반복측정(repeated measures) 구조 추가
     - 같은 도시의 12개 station은 사실상 "하나의 대기질을 12번 관측"한 것에 가깝다.
     - 날짜별 도시 공통 랜덤효과 u[t]를 추가하여, 같은 날 12개 관측이 공유하는 성분을 분리:
         mu[i,t] = alpha_station[i] + X[i,t]·beta_station[i] + u[t]
         u[t] ~ Normal(0, sigma_day)
     - 이로써 유효표본크기가 "17,000 station-day"가 아니라 "약 1,460일"에 가깝게 조정되어
       기상 계수(beta)의 신용구간이 정직하게 넓어진다.

PRIOR SPECIFICATION (C0)
------------------------
  gamma_group[g]   ~ Normal(4.0, 1.0)      : 그룹별 절편(고정효과). log1p(PM2.5)의 중심이
                                             대략 4 (PM2.5 ~ 54 ug/m3)이고, sd=1이면
                                             PM2.5 약 20~150 범위를 덮는 약정보 사전분포.
  tau_station      ~ HalfNormal(0.5)       : station 간 절편 편차. log 스케일에서 0.5는
                                             station 간 농도비 약 1.6배까지 허용.
  station_offset   ~ Normal(0, 1)          : 비중심화(non-centered) 보조변수.
  beta_global[k]   ~ Normal(0, 1)          : 표준화된 기상변수의 전역 계수. 1 sd 변화가
                                             log(PM2.5)를 1 이상 바꾸는 일은 드물다는 사전지식.
  tau_beta[k]      ~ HalfNormal(0.5)       : 기상 반응의 station 간 이질성.
  beta_offset      ~ Normal(0, 1)          : 비중심화 보조변수.
  sigma_day        ~ HalfNormal(0.5)       : 도시 공통 일별 변동의 크기.
  u_raw[t]         ~ Normal(0, 1)          : 일별 랜덤효과 비중심화 보조변수.
  rho              ~ TruncatedNormal(0.5, 0.3, -0.95, 0.95)
                                           : AR(1) 계수. 대기오염 일별 자기상관이 양이고
                                             0.4~0.8 부근이라는 사전지식을 약하게 반영.
  sigma_e          ~ HalfNormal(1.0)       : station-day 고유 잔차 표준편차.

입력
----
01_preprocess.py 로 만든 전처리 완료 파일(cleaned_prsa_all.csv).

출력 (OUTPUT_DIR 아래)
----------------------
- prior_predictive_check.png    : 사전예측검정 (C0)
- model_comparison.csv          : pooled / unpooled / hierarchical LOO 비교
- posterior_summary.csv         : station별 파라미터 사후분포 요약
- convergence_diagnostics.csv   : r_hat / ESS 명시적 산출
- station_intercepts_forest.png
- ppc_check.png
- trace_diagnostics.png
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"   # Mac: "AppleGothic", Colab: "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------
INPUT_PATH = r"불러올 전처리 파일 경로(파일명.확장자까지)"
OUTPUT_DIR = r"저장할 위치"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRAWS = 1000
TUNE = 1000
CHAINS = 4              # 로컬 코어가 4개 이상이면 4 권장 (r_hat 신뢰도 확보)
TARGET_ACCEPT = 0.9
RANDOM_SEED = 42

# 구조 옵션 (ablation 실험용 스위치)
USE_AR1 = True          # C2: 잔차 AR(1)
USE_DAY_EFFECT = True   # C3: 날짜별 도시 공통 랜덤효과

# 모델 비교(pooled/unpooled 추가 샘플링)는 시간이 3배 든다. 필요할 때만 True.
RUN_MODEL_COMPARISON = True

STATION_GROUP = {
    "Aotizhongxin": "urban", "Dongsi": "urban", "Guanyuan": "urban",
    "Wanliu": "urban", "Wanshouxigong": "urban", "Nongzhanguan": "urban",
    "Tiantan": "urban",
    "Gucheng": "suburban_industrial", "Shunyi": "suburban_industrial",
    "Changping": "suburban_industrial",
    "Dingling": "background", "Huairou": "background",
}

FEATURES = ["TEMP", "PRES", "DEWP", "WSPM"]


# ----------------------------------------------------------------------
# 1. 데이터 준비
# ----------------------------------------------------------------------
def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["group"] = df["station"].map(STATION_GROUP)

    daily = (
        df.groupby(["station", "group", pd.Grouper(key="datetime", freq="D")])
        .agg({**{f: "mean" for f in FEATURES}, "PM2.5": "mean", "RAIN": "sum"})
        .reset_index()
        .rename(columns={"datetime": "date"})
    )
    daily = daily.dropna(subset=["PM2.5"] + FEATURES)
    daily["log_pm25"] = np.log1p(daily["PM2.5"])

    for f in FEATURES:
        daily[f + "_z"] = (daily[f] - daily[f].mean()) / daily[f].std()

    # AR(1) 계산을 위해 station -> 날짜 순으로 정렬 (인덱스 재부여 필수)
    daily = daily.sort_values(["station", "date"]).reset_index(drop=True)
    return daily


def build_ar_index(daily, station_col="station", time_col="date", unit="D"):
    """
    각 행에 대해 '같은 station의 직전 시점 행 인덱스'와 '직전 시점이 실제로 존재하는지' 플래그를 만든다.
    날짜가 중간에 끊긴 경우(결측일)는 AR 체인을 끊어 새로운 시계열 시작으로 취급한다.
    """
    n = len(daily)
    prev_idx = np.arange(n)          # 직전이 없으면 자기 자신(인덱싱 안전용)
    has_prev = np.zeros(n)

    for _, g in daily.groupby(station_col):
        idx = g.index.values
        if unit == "D":
            gap = g[time_col].diff().dt.days.values
        else:                        # 월 인덱스(정수)로 들어오는 경우
            gap = g[time_col].diff().values
        for j in range(1, len(idx)):
            if gap[j] == 1:
                prev_idx[idx[j]] = idx[j - 1]
                has_prev[idx[j]] = 1.0
    return prev_idx, has_prev


def build_indices(daily):
    stations = sorted(daily["station"].unique())
    groups = sorted(daily["group"].unique())
    station_idx = daily["station"].map({s: i for i, s in enumerate(stations)}).values
    group_of_station = np.array([groups.index(STATION_GROUP[s]) for s in stations])

    dates = np.sort(daily["date"].unique())
    # datetime64 <-> Timestamp 타입 불일치를 피하려고 get_indexer 사용
    date_idx = pd.DatetimeIndex(dates).get_indexer(pd.DatetimeIndex(daily["date"]))
    return stations, groups, station_idx, group_of_station, dates, date_idx


# ----------------------------------------------------------------------
# 2. 모델 정의
#    kind = "hierarchical" | "pooled" | "unpooled"
#    세 모델 모두 동일한 시간구조(AR(1) + 일별 랜덤효과)를 갖게 하여,
#    비교의 초점이 오직 "station 파라미터의 풀링 방식"에만 맞춰지도록 했다.
# ----------------------------------------------------------------------
def build_model(daily, stations, groups, station_idx, group_of_station,
                dates, date_idx, prev_idx, has_prev, kind="hierarchical"):
    X = daily[[f + "_z" for f in FEATURES]].values
    y = daily["log_pm25"].values

    coords = {
        "station": stations,
        "group": groups,
        "feature": FEATURES,
        "date": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "obs_id": np.arange(len(daily)),
    }

    with pm.Model(coords=coords) as model:
        station_idx_ = pm.Data("station_idx", station_idx, dims="obs_id")
        date_idx_ = pm.Data("date_idx", date_idx, dims="obs_id")
        prev_idx_ = pm.Data("prev_idx", prev_idx, dims="obs_id")
        has_prev_ = pm.Data("has_prev", has_prev, dims="obs_id")
        y_prev_ = pm.Data("y_prev", y[prev_idx], dims="obs_id")
        X_ = pm.Data("X", X, dims=("obs_id", "feature"))
        y_ = pm.Data("y_obs", y, dims="obs_id")

        # ---------------- 절편 구조 ----------------
        if kind == "hierarchical":
            # [C1] 그룹 효과는 고정효과(각 그룹에 독립적인 약정보 prior)
            gamma_group = pm.Normal("gamma_group", mu=4.0, sigma=1.0, dims="group")
            # station 절편만 그룹 평균으로 부분 풀링 (비중심화)
            tau_station = pm.HalfNormal("tau_station", sigma=0.5)
            station_offset = pm.Normal("station_offset", 0, 1, dims="station")
            alpha_station = pm.Deterministic(
                "alpha_station",
                gamma_group[group_of_station] + tau_station * station_offset,
                dims="station",
            )
            beta_global = pm.Normal("beta_global", 0, 1, dims="feature")
            tau_beta = pm.HalfNormal("tau_beta", sigma=0.5, dims="feature")
            beta_offset = pm.Normal("beta_offset", 0, 1, dims=("station", "feature"))
            beta_station = pm.Deterministic(
                "beta_station", beta_global + tau_beta * beta_offset,
                dims=("station", "feature"),
            )
            alpha_obs = alpha_station[station_idx_]
            beta_obs = beta_station[station_idx_]

        elif kind == "unpooled":
            alpha_station = pm.Normal("alpha_station", 4.0, 1.0, dims="station")
            beta_station = pm.Normal("beta_station", 0, 1, dims=("station", "feature"))
            alpha_obs = alpha_station[station_idx_]
            beta_obs = beta_station[station_idx_]

        elif kind == "pooled":
            alpha = pm.Normal("alpha", 4.0, 1.0)
            beta = pm.Normal("beta", 0, 1, dims="feature")
            alpha_obs = alpha
            beta_obs = beta[None, :]

        else:
            raise ValueError(kind)

        mu = alpha_obs + (X_ * beta_obs).sum(axis=-1)

        # ---------------- [C3] 날짜별 도시 공통 랜덤효과 ----------------
        if USE_DAY_EFFECT:
            sigma_day = pm.HalfNormal("sigma_day", sigma=0.5)
            u_raw = pm.Normal("u_raw", 0, 1, dims="date")
            u = sigma_day * u_raw
            mu = mu + u[date_idx_]

        # ---------------- [C2] 잔차 AR(1) ----------------
        sigma_e = pm.HalfNormal("sigma_e", sigma=1.0)
        if USE_AR1:
            rho = pm.TruncatedNormal("rho", mu=0.5, sigma=0.3, lower=-0.95, upper=0.95)
            mu_prev = mu[prev_idx_]
            mu_cond = mu + rho * has_prev_ * (y_prev_ - mu_prev)
            sigma_vec = pt.switch(
                pt.gt(has_prev_, 0.5), sigma_e, sigma_e / pt.sqrt(1.0 - rho ** 2)
            )
        else:
            mu_cond = mu
            sigma_vec = sigma_e

        pm.Normal("obs", mu=mu_cond, sigma=sigma_vec, observed=y_, dims="obs_id")

    return model


# ----------------------------------------------------------------------
# 3. 진단 유틸
# ----------------------------------------------------------------------
def diagnostics(idata, var_names, tag, output_dir):
    """az.summary 의 r_hat 이 비는 환경을 대비해 az.rhat / az.ess 로 명시 산출 (v1 한계 보완)."""
    rhat_ds = az.rhat(idata, var_names=var_names)
    ess_ds = az.ess(idata, var_names=var_names)

    rows = []
    for v in rhat_ds.data_vars:
        rows.append({"variable": v,
                     "rhat_max": float(rhat_ds[v].max()),
                     "ess_min": float(ess_ds[v].min())})
    diag = pd.DataFrame(rows)
    diag.to_csv(os.path.join(output_dir, f"convergence_diagnostics_{tag}.csv"), index=False)

    max_rhat = float(diag["rhat_max"].max())
    min_ess = float(diag["ess_min"].min())
    print(f"  [{tag}] r_hat 최대 {max_rhat:.4f} / ESS 최소 {min_ess:.0f}")
    if max_rhat > 1.01:
        print("  [경고] r_hat > 1.01 -> tune/draws 증가 후 재실행 권장")
    if min_ess < 400:
        print("  [경고] ESS < 400 -> draws 증가 권장")
    return max_rhat, min_ess


# ----------------------------------------------------------------------
# 4. 실행
# ----------------------------------------------------------------------
def main():
    print("[1/7] 데이터 로드 및 일단위 집계...")
    daily = load_and_prepare(INPUT_PATH)
    stations, groups, station_idx, group_of_station, dates, date_idx = build_indices(daily)
    prev_idx, has_prev = build_ar_index(daily)
    print(f"  -> {len(daily)} station-day rows, {len(stations)} stations, "
          f"{len(groups)} groups, {len(dates)} 일")
    print(f"  -> AR(1) 체인 연결 비율: {has_prev.mean():.1%} "
          f"(나머지는 시계열 시작 또는 결측일로 인한 단절)")

    common = dict(daily=daily, stations=stations, groups=groups,
                  station_idx=station_idx, group_of_station=group_of_station,
                  dates=dates, date_idx=date_idx, prev_idx=prev_idx, has_prev=has_prev)

    sample_kwargs = dict(
        draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
        target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True,
    )

    print("[2/7] 사전예측검정(prior predictive check)...")
    hier_model = build_model(kind="hierarchical", **common)
    with hier_model:
        prior = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)
    prior_y = prior.prior_predictive["obs"].values.flatten()
    plt.figure(figsize=(7, 4.5))
    plt.hist(prior_y, bins=80, density=True, alpha=0.5, label="사전예측분포")
    plt.hist(daily["log_pm25"], bins=80, density=True, alpha=0.5, label="관측값")
    plt.xlabel("log1p(PM2.5)")
    plt.title("사전예측검정: 사전분포가 관측 범위를 충분히 덮는가")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prior_predictive_check.png"), dpi=150)
    plt.close()
    print(f"  -> 사전예측 y 범위 [{np.percentile(prior_y, 1):.2f}, {np.percentile(prior_y, 99):.2f}], "
          f"관측 범위 [{daily['log_pm25'].min():.2f}, {daily['log_pm25'].max():.2f}]")

    print("[3/7] 계층모델(hierarchical) 샘플링...")
    with hier_model:
        idata_hier = pm.sample(**sample_kwargs)
        pm.compute_log_likelihood(idata_hier)
        pm.sample_posterior_predictive(idata_hier, extend_inferencedata=True,
                                       random_seed=RANDOM_SEED)

    print("[4/7] 수렴 진단...")
    diag_vars = ["gamma_group", "tau_station", "beta_global", "tau_beta", "sigma_e"]
    if USE_DAY_EFFECT:
        diag_vars.append("sigma_day")
    if USE_AR1:
        diag_vars.append("rho")
    diagnostics(idata_hier, diag_vars, "hierarchical", OUTPUT_DIR)
    print(az.summary(idata_hier, var_names=diag_vars))

    az.plot_trace(idata_hier, var_names=diag_vars)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trace_diagnostics.png"), dpi=150)
    plt.close()

    if USE_AR1:
        rho_s = idata_hier.posterior["rho"].values.flatten()
        print(f"  -> AR(1) 계수 rho 사후평균 {rho_s.mean():.3f} "
              f"(94% HDI {np.percentile(rho_s, 3):.3f} ~ {np.percentile(rho_s, 97):.3f})")
    if USE_DAY_EFFECT:
        sd_s = idata_hier.posterior["sigma_day"].values.flatten()
        se_s = idata_hier.posterior["sigma_e"].values.flatten()
        print(f"  -> sigma_day {sd_s.mean():.3f} vs sigma_e {se_s.mean():.3f} "
              f"(비율 {(sd_s.mean()**2/(sd_s.mean()**2+se_s.mean()**2)):.1%}"
              f" = 같은 날 12개 station이 공유하는 분산 비중)")

    print("[5/7] 모델 비교 (LOO)...")
    if RUN_MODEL_COMPARISON:
        pooled_model = build_model(kind="pooled", **common)
        with pooled_model:
            idata_pooled = pm.sample(**sample_kwargs)
            pm.compute_log_likelihood(idata_pooled)

        unpooled_model = build_model(kind="unpooled", **common)
        with unpooled_model:
            idata_unpooled = pm.sample(**sample_kwargs)
            pm.compute_log_likelihood(idata_unpooled)

        try:
            cmp_df = az.compare({
                "hierarchical": idata_hier,
                "pooled": idata_pooled,
                "unpooled": idata_unpooled,
            })
            cmp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
            print(cmp_df)
            print("  [주의] AR(1) 조건부 우도 기반 pointwise LOO 이므로 엄밀한 시계열 교차검증은 아님.")
        except Exception as e:
            print(f"  [주의] LOO 비교 실패: {e}")
    else:
        print("  (RUN_MODEL_COMPARISON=False 로 생략)")

    print("[6/7] 시각화...")
    az.plot_forest(idata_hier, var_names=["alpha_station"], combined=True)
    plt.title("Station별 log(PM2.5) 절편 사후분포 (94% HDI)\nAR(1)+일별 랜덤효과 반영")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "station_intercepts_forest.png"), dpi=150)
    plt.close()

    az.plot_forest(idata_hier, var_names=["beta_global", "gamma_group"], combined=True)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("전역 기상계수 및 그룹 고정효과 (94% HDI)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "beta_group_forest.png"), dpi=150)
    plt.close()

    try:
        az.plot_ppc(idata_hier, num_pp_samples=100)
    except AttributeError:
        az.plot_ppc_dist(idata_hier, num_samples=100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ppc_check.png"), dpi=150)
    plt.close()

    print("[7/7] 결과 저장...")
    full_summary = az.summary(
        idata_hier, var_names=["alpha_station", "gamma_group", "beta_station", "beta_global"]
    )
    full_summary.to_csv(os.path.join(OUTPUT_DIR, "posterior_summary.csv"))
    print(f"완료. 결과는 {OUTPUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
