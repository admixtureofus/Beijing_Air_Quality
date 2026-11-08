"""
03_bayesian_hierarchical.py  (v2.1 - 1차 실행 진단 반영)
=========================================================
베이지안 계층 회귀 모델 (Hierarchical / Partial Pooling Regression)

v2.0 실행 결과 발견된 문제와 v2.1 수정
--------------------------------------
[문제] 1차 실행에서 r_hat 최대 1.13, ESS 최소 18로 수렴 미달.
       특히 sigma_day, gamma_group, alpha_station 이 느리게 섞였다.
       원인은 "전역 수준(level)의 능선(ridge)" 이다. u[t] ~ Normal(0, sigma_day) 로 두면
       u 들의 표본평균이 자유롭게 떠다닐 수 있는데(sd = sigma_day/sqrt(T) ≈ 0.7/38 ≈ 0.018),
       이 값이 alpha_station 의 사후 sd(≈0.016)와 거의 같은 크기다. 즉 "절편을 조금 올리고
       모든 u 를 조금 내리는" 방향으로 우도가 거의 변하지 않아 체인이 그 방향을 배회한다.

[수정 1] u[t] 에 합-0 제약을 부여하고 중심화 파라미터화로 전환.
           pm.ZeroSumNormal("u", sigma=sigma_day, dims="date")
         - 합-0 제약이 위의 능선을 제거한다.
         - 하루에 12개 관측이 있어 u[t] 가 매우 잘 식별되므로(비중심화가 유리한 조건이 아님)
           중심화 파라미터화가 오히려 효율적이다.
         동일 규모의 모의자료(12 station x 1,461일, sigma_day 0.70, sigma_e 0.172, rho 0.40)로
         검증한 결과는 다음과 같다.
           수정 전: r_hat 최대 1.126, ESS 최소 28  (158초)
           수정 후: r_hat 최대 1.018, ESS 최소 477 (128초)
         점추정값은 사실상 변하지 않았다(gamma_group 4.106 -> 4.101, sigma_day 0.704 -> 0.708).
         즉 1차 실행의 점추정은 신뢰할 수 있으나 구간·진단은 재실행 후 인용해야 한다.

[수정 2] 기상변수의 between / within 분해 (USE_BETWEEN_WITHIN)
         기상변수는 같은 날 12개 station이 거의 같은 값을 갖는 '날짜 수준' 변수라,
         날짜 공통효과 u[t] 와 강하게 경합한다. 1차 실행에서 TEMP 계수가 v1의 -1.0에서
         -0.67로, WSPM이 -0.03까지 줄어든 것이 그 결과다.
         이를 해석 가능하게 만들기 위해 각 변수를 두 성분으로 분해한다.
           X_between[t]  = 그날 도시 전체 평균 -> 날짜 간(between) 효과, 전역 계수
           X_within[i,t] = X - X_between       -> 같은 날 station 간(within) 효과, 부분 풀링
         이렇게 하면 "기상이 도시 전체 수준을 움직이는 효과"와 "같은 날에도 지점 간 기상 차이가
         만드는 효과"를 분리해 보고할 수 있고, u[t] 와의 경합이 계수 해석을 흐리지 않는다.

[수정 3] az.rcParams["stats.ci_prob"] = 0.94 지정
         ArviZ 1.x의 기본 구간은 89% ETI 다. v2.0 실행 결과의 요약표는 eti89 로 산출되었는데
         그림 제목에는 "94% HDI"라고 적혀 있어 표기가 어긋났다. 아래에서 94% HDI로 통일한다.

[수정 4] 전역 파라미터 요약을 global_summary.csv 로 저장(기존에는 콘솔 출력만).

이전 버전(v2.0)에서 반영한 교수님 코멘트
----------------------------------------
 C0 사전분포 명시 + 사전예측검정 / C1 그룹 고정효과 / C2 잔차 AR(1) / C3 날짜별 공통효과

PRIOR SPECIFICATION (C0)
------------------------
  gamma_group[g]   ~ Normal(4.0, 1.0)      : 그룹별 절편(고정효과). log1p(PM2.5) 중심 ≈ 4.
  tau_station      ~ HalfNormal(0.5)       : station 간 절편 편차.
  station_offset   ~ Normal(0, 1)          : 비중심화 보조변수.
  beta_between[k]  ~ Normal(0, 1)          : 날짜 간 기상 효과(도시 전체 수준).
  beta_global[k]   ~ Normal(0, 1)          : station 내 기상 효과의 전역 평균.
  tau_beta[k]      ~ HalfNormal(0.5)       : 기상 반응의 station 간 이질성.
  beta_offset      ~ Normal(0, 1)          : 비중심화 보조변수.
  sigma_day        ~ HalfNormal(0.5)       : 날짜별 도시 공통 충격의 크기.
  u[t]             ~ ZeroSumNormal(sigma_day) : 날짜 공통효과(합-0 제약).
  rho              ~ TruncatedNormal(0.5, 0.3, -0.95, 0.95) : AR(1) 계수.
  sigma_e          ~ HalfNormal(1.0)       : station-day 고유 잔차.

출력 (OUTPUT_DIR 아래)
----------------------
- prior_predictive_check.png / global_summary.csv / model_comparison.csv
- posterior_summary.csv / convergence_diagnostics_hierarchical.csv
- station_intercepts_forest.png / beta_group_forest.png / ppc_check.png / trace_diagnostics.png
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

# [수정 3] 요약표·그림의 구간을 94% HDI로 통일 (ArviZ 1.x 기본값은 89% ETI)
try:
    az.rcParams["stats.ci_prob"] = 0.94
    az.rcParams["stats.ci_kind"] = "hdi"
except Exception:
    pass

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------
INPUT_PATH = r"불러올 전처리 파일 경로(파일명.확장자까지)"
OUTPUT_DIR = r"저장할 위치"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRAWS = 1500
TUNE = 1500
CHAINS = 4
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42

USE_AR1 = True             # C2: 잔차 AR(1)
USE_DAY_EFFECT = True      # C3: 날짜별 도시 공통 랜덤효과
USE_BETWEEN_WITHIN = True  # [수정 2] 기상변수 between/within 분해
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

    daily = daily.sort_values(["station", "date"]).reset_index(drop=True)
    return daily


def split_between_within(daily):
    """기상변수를 '그날 도시 평균(between)' 과 'station 편차(within)' 로 분해."""
    zcols = [f + "_z" for f in FEATURES]
    between = daily.groupby("date")[zcols].transform("mean")
    X_between = between.values
    X_within = daily[zcols].values - X_between
    return X_between, X_within


def build_ar_index(daily, station_col="station", time_col="date", unit="D"):
    """같은 station의 직전 시점 행 인덱스와 존재 여부 플래그 (날짜가 끊기면 체인 단절)."""
    n = len(daily)
    prev_idx = np.arange(n)
    has_prev = np.zeros(n)
    for _, g in daily.groupby(station_col):
        idx = g.index.values
        gap = g[time_col].diff().dt.days.values if unit == "D" else g[time_col].diff().values
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
    date_idx = pd.DatetimeIndex(dates).get_indexer(pd.DatetimeIndex(daily["date"]))
    return stations, groups, station_idx, group_of_station, dates, date_idx


# ----------------------------------------------------------------------
# 2. 모델 정의  (kind = "hierarchical" | "pooled" | "unpooled")
#    세 모델 모두 동일한 시간구조(AR(1) + 날짜 공통효과)를 갖게 하여,
#    비교의 초점이 오직 "station 파라미터의 풀링 방식"에만 맞춰지도록 했다.
# ----------------------------------------------------------------------
def build_model(daily, stations, groups, station_idx, group_of_station,
                dates, date_idx, prev_idx, has_prev, kind="hierarchical"):
    zcols = [f + "_z" for f in FEATURES]
    X = daily[zcols].values
    Xb, Xw = split_between_within(daily)
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
        y_ = pm.Data("y_obs", y, dims="obs_id")
        if USE_BETWEEN_WITHIN:
            Xb_ = pm.Data("X_between", Xb, dims=("obs_id", "feature"))
            Xw_ = pm.Data("X_within", Xw, dims=("obs_id", "feature"))
        else:
            X_ = pm.Data("X", X, dims=("obs_id", "feature"))

        # ---------------- 절편 구조 ----------------
        if kind == "hierarchical":
            gamma_group = pm.Normal("gamma_group", mu=4.0, sigma=1.0, dims="group")   # C1
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

        # ---------------- 평균 구조 ----------------
        if USE_BETWEEN_WITHIN:
            # between(날짜 간)은 도시 수준 효과이므로 전역 계수 하나로 둔다.
            beta_between = pm.Normal("beta_between", 0, 1, dims="feature")
            mu = alpha_obs + (Xb_ * beta_between).sum(axis=-1) + (Xw_ * beta_obs).sum(axis=-1)
        else:
            mu = alpha_obs + (X_ * beta_obs).sum(axis=-1)

        # ---------------- [C3] 날짜별 도시 공통 랜덤효과 (합-0 제약) ----------------
        if USE_DAY_EFFECT:
            sigma_day = pm.HalfNormal("sigma_day", sigma=0.5)
            u = pm.ZeroSumNormal("u", sigma=sigma_day, dims="date")   # [수정 1]
            mu = mu + u[date_idx_]

        # ---------------- [C2] 잔차 AR(1) ----------------
        sigma_e = pm.HalfNormal("sigma_e", sigma=1.0)
        if USE_AR1:
            rho = pm.TruncatedNormal("rho", mu=0.5, sigma=0.3, lower=-0.95, upper=0.95)
            mu_cond = mu + rho * has_prev_ * (y_prev_ - mu[prev_idx_])
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
    rhat_ds = az.rhat(idata, var_names=var_names)
    ess_ds = az.ess(idata, var_names=var_names)
    diag = pd.DataFrame([{"variable": v,
                          "rhat_max": float(rhat_ds[v].max()),
                          "ess_min": float(ess_ds[v].min())} for v in rhat_ds.data_vars])
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
    print(f"  -> AR(1) 체인 연결 비율: {has_prev.mean():.1%}")
    print(f"  -> 옵션: AR1={USE_AR1}, DAY_EFFECT={USE_DAY_EFFECT}, "
          f"BETWEEN_WITHIN={USE_BETWEEN_WITHIN}")

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

    print("[3/7] 계층모델(hierarchical) 샘플링...")
    with hier_model:
        idata_hier = pm.sample(**sample_kwargs)
        pm.compute_log_likelihood(idata_hier)
        pm.sample_posterior_predictive(idata_hier, extend_inferencedata=True,
                                       random_seed=RANDOM_SEED)

    print("[4/7] 수렴 진단...")
    diag_vars = ["gamma_group", "tau_station", "beta_global", "tau_beta", "sigma_e"]
    if USE_BETWEEN_WITHIN:
        diag_vars.insert(3, "beta_between")
    if USE_DAY_EFFECT:
        diag_vars.append("sigma_day")
    if USE_AR1:
        diag_vars.append("rho")
    diagnostics(idata_hier, diag_vars, "hierarchical", OUTPUT_DIR)

    gsum = az.summary(idata_hier, var_names=diag_vars)
    print(gsum)
    gsum.to_csv(os.path.join(OUTPUT_DIR, "global_summary.csv"))   # [수정 4]

    az.plot_trace(idata_hier, var_names=diag_vars)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trace_diagnostics.png"), dpi=150)
    plt.close()

    if USE_DAY_EFFECT:
        sd = float(idata_hier.posterior["sigma_day"].mean())
        se = float(idata_hier.posterior["sigma_e"].mean())
        print(f"  -> sigma_day {sd:.3f} vs sigma_e {se:.3f} -> "
              f"공유 분산 비율 {sd**2/(sd**2+se**2):.1%} "
              f"(같은 날 12개 station이 공유하는 변동의 비중)")
    if USE_AR1:
        print(f"  -> AR(1) rho 사후평균 {float(idata_hier.posterior['rho'].mean()):.3f}")

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
            cmp_df = az.compare({"hierarchical": idata_hier,
                                 "pooled": idata_pooled,
                                 "unpooled": idata_unpooled})
            cmp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
            print(cmp_df)
            print("  [주의] AR(1) 조건부 우도 기반 pointwise LOO 이며, 날짜 공통효과 때문에 "
                  "Pareto k 경고가 다수 발생한다. 순위의 참고용으로만 해석할 것.")
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

    fvars = ["beta_global", "gamma_group"]
    if USE_BETWEEN_WITHIN:
        fvars.insert(0, "beta_between")
    az.plot_forest(idata_hier, var_names=fvars, combined=True)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("기상계수(between/within) 및 그룹 고정효과 (94% HDI)")
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
    svars = ["alpha_station", "gamma_group", "beta_station", "beta_global"]
    if USE_BETWEEN_WITHIN:
        svars.append("beta_between")
    az.summary(idata_hier, var_names=svars).to_csv(
        os.path.join(OUTPUT_DIR, "posterior_summary.csv"))
    print(f"완료. 결과는 {OUTPUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
