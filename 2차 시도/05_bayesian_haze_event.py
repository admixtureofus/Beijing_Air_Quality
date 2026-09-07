"""
05_bayesian_haze_event.py  (v2 - 지도교수 코멘트 반영본)
========================================================
베이지안 계층 로지스틱 회귀 - 고농도(haze) 이벤트 확률 모델

v1 대비 변경점 (교수님 코멘트 반영)
-----------------------------------
[C0] 사전분포 명시 + 사전예측검정 수행 (아래 PRIOR SPECIFICATION 참조).

[C1] 그룹 효과를 random -> fixed
     v1은 mu_group ~ Normal(mu_global, tau_group) 로 3개 그룹 절편을 랜덤효과로 두었으나,
     그룹이 3개뿐이라 tau_group은 데이터로 식별되지 않는다.
     v2는 gamma_group ~ Normal(-2.0, 1.5) 를 그룹마다 독립 부여(고정효과)하고,
     station 절편만 그룹 평균 주위로 부분 풀링한다.

[C2] 시간 상관 반영
     Bernoulli 우도에서는 "잔차의 AR(1)"이 그대로 정의되지 않으므로, 동등한 역할을
     (a) lag_logpm25 (전날 농도) 공변량과
     (b) 날짜별 공통 랜덤효과 u[t] 로 나누어 처리했다.
     즉 관측치를 독립으로 두지 않고, 같은 날/이웃 날의 상관을 명시적으로 모델에 넣는다.

[C3] 반복측정 구조 반영
     12개 station은 같은 도시의 대기를 12번 관측한 것에 가깝다.
     날짜별 공통 랜덤효과 u[t] ~ Normal(0, sigma_day) 를 logit 에 더해,
     같은 날 12개 관측이 공유하는 충격을 분리한다. 이로써 계수의 신용구간이
     "17,000 관측"이 아니라 "약 1,460일"에 걸맞게 넓어진다.

[C4] 시계열 홀드아웃 평가
     랜덤 분할 대신 마지막 12개월(TEST_MONTHS)을 테스트셋으로 고정했다.
     - 표준화(z-score) 통계량은 학습기간에서만 계산해 테스트에 적용(정보 누수 방지).
     - 테스트 예측확률은 미래 날짜의 u[t]를 알 수 없으므로 사전분포에서 적분한
       주변예측확률(marginal predictive probability)로 계산했다.
       공정한 비교를 위해 학습기간 성능도 동일한 주변예측 방식으로 산출한다.

PRIOR SPECIFICATION (C0)
------------------------
  gamma_group[g] ~ Normal(-2.0, 1.5)   : 그룹별 기저 로짓(고정효과). logit(-2) ≈ 12%로
                                         관측된 이벤트율 근방을 중심에 둔 약정보 사전분포.
  tau_station    ~ HalfNormal(0.5)     : 그룹 내 station 절편 편차.
  station_offset ~ Normal(0, 1)        : 비중심화 보조변수.
  beta_global[k] ~ Normal(0, 1)        : 표준화 변수의 전역 로짓 계수(1 sd 변화당 오즈 e^±1 근방).
  tau_beta[k]    ~ HalfNormal(0.5)     : 계수의 station 간 이질성.
  beta_offset    ~ Normal(0, 1)        : 비중심화 보조변수.
  sigma_day      ~ HalfNormal(1.0)     : 날짜별 공통 충격의 크기 (C2/C3).
  u_raw[t]       ~ Normal(0, 1)        : 비중심화 보조변수.

출력 (OUTPUT_DIR 아래)
----------------------
- haze_prior_predictive.png
- haze_event_summary.csv / haze_convergence.csv
- haze_trace_diagnostics.png
- haze_forest_alpha.png / haze_forest_beta.png
- haze_calibration_train.png / haze_calibration_test.png
- haze_model_metrics.txt  (학습기간 vs 홀드아웃 1년 성능)
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

plt.rcParams["font.family"] = "Malgun Gothic"   # Mac: AppleGothic
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------
INPUT_PATH = r"파일경로\cleaned_prsa_all.csv"
OUTPUT_DIR = r"저장위치/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENT_THRESHOLD = 150.0   # 일평균 PM2.5 (ug/m3)
TEST_MONTHS = 12          # [C4] 마지막 12개월을 홀드아웃

DRAWS = 1500
TUNE = 1500
CHAINS = 4
TARGET_ACCEPT = 0.9
RANDOM_SEED = 42
N_PRED_DRAWS = 500        # 주변예측확률 계산에 사용할 사후표본 수

USE_DAY_EFFECT = True     # C2/C3

STATION_GROUP = {
    "Aotizhongxin": "urban", "Dongsi": "urban", "Guanyuan": "urban",
    "Wanliu": "urban", "Wanshouxigong": "urban", "Nongzhanguan": "urban",
    "Tiantan": "urban",
    "Gucheng": "suburban_industrial", "Shunyi": "suburban_industrial",
    "Changping": "suburban_industrial",
    "Dingling": "background", "Huairou": "background",
}

RAW_FEATURES = ["TEMP", "PRES", "DEWP", "WSPM", "RAIN", "lag_logpm25"]
FEATURES = [f + "_z" for f in RAW_FEATURES] + ["sin_doy", "cos_doy"]


# ----------------------------------------------------------------------
# 1. 데이터 준비
# ----------------------------------------------------------------------
def load_daily_events(path):
    df = pd.read_csv(path, parse_dates=["datetime"])
    for c in ["PM2.5", "TEMP", "PRES", "DEWP", "WSPM", "RAIN"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = df["datetime"].dt.normalize()
    daily = (
        df.groupby(["station", "date"])
        .agg(PM25=("PM2.5", "mean"), TEMP=("TEMP", "mean"), PRES=("PRES", "mean"),
             DEWP=("DEWP", "mean"), WSPM=("WSPM", "mean"), RAIN=("RAIN", "sum"))
        .reset_index()
    )
    daily["group"] = daily["station"].map(STATION_GROUP)
    daily["event"] = (daily["PM25"] > EVENT_THRESHOLD).astype(int)
    daily["log_pm25"] = np.log1p(daily["PM25"])

    daily = daily.sort_values(["station", "date"])
    daily["lag_logpm25"] = daily.groupby("station")["log_pm25"].shift(1)

    doy = daily["date"].dt.dayofyear
    daily["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    daily["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    daily = daily.dropna(subset=["event"] + RAW_FEATURES).reset_index(drop=True)
    return daily


def time_split(daily, test_months=TEST_MONTHS):
    """[C4] 랜덤 분할이 아니라 마지막 test_months 개월을 테스트로 사용."""
    cutoff = daily["date"].max() - pd.DateOffset(months=test_months)
    train = daily[daily["date"] <= cutoff].copy().reset_index(drop=True)
    test = daily[daily["date"] > cutoff].copy().reset_index(drop=True)
    return train, test, cutoff


def standardize(train, test):
    """표준화 통계량은 학습기간에서만 계산 (정보 누수 방지)."""
    stats = {}
    for f in RAW_FEATURES:
        m, s = train[f].mean(), train[f].std()
        stats[f] = (m, s)
        train[f + "_z"] = (train[f] - m) / s
        test[f + "_z"] = (test[f] - m) / s
    return train, test, stats


def build_indices(train):
    stations = sorted(train["station"].unique())
    groups = sorted(train["group"].unique())
    station_idx = train["station"].map({s: i for i, s in enumerate(stations)}).values
    group_of_station = np.array([groups.index(STATION_GROUP[s]) for s in stations])
    dates = np.sort(train["date"].unique())
    date_idx = pd.DatetimeIndex(dates).get_indexer(pd.DatetimeIndex(train["date"]))
    return stations, groups, station_idx, group_of_station, dates, date_idx


# ----------------------------------------------------------------------
# 2. 모델 정의
# ----------------------------------------------------------------------
def build_model(train, stations, groups, station_idx, group_of_station, dates, date_idx):
    X = train[FEATURES].values
    y = train["event"].values

    coords = {
        "station": stations, "group": groups, "feature": FEATURES,
        "date": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "obs_id": np.arange(len(train)),
    }

    with pm.Model(coords=coords) as model:
        station_idx_ = pm.Data("station_idx", station_idx, dims="obs_id")
        date_idx_ = pm.Data("date_idx", date_idx, dims="obs_id")
        X_ = pm.Data("X", X, dims=("obs_id", "feature"))
        y_ = pm.Data("y_obs", y, dims="obs_id")

        # -- [C1] 그룹 = 고정효과, station 만 부분 풀링 --
        gamma_group = pm.Normal("gamma_group", mu=-2.0, sigma=1.5, dims="group")
        tau_station = pm.HalfNormal("tau_station", sigma=0.5)
        station_offset = pm.Normal("station_offset", 0, 1, dims="station")
        alpha_station = pm.Deterministic(
            "alpha_station",
            gamma_group[group_of_station] + tau_station * station_offset,
            dims="station",
        )

        # -- 계수: station별 부분 풀링 --
        beta_global = pm.Normal("beta_global", 0, 1, dims="feature")
        tau_beta = pm.HalfNormal("tau_beta", sigma=0.5, dims="feature")
        beta_offset = pm.Normal("beta_offset", 0, 1, dims=("station", "feature"))
        beta_station = pm.Deterministic(
            "beta_station", beta_global + tau_beta * beta_offset,
            dims=("station", "feature"),
        )

        logit_p = alpha_station[station_idx_] + (X_ * beta_station[station_idx_]).sum(axis=-1)

        # -- [C2][C3] 날짜별 도시 공통 랜덤효과 --
        if USE_DAY_EFFECT:
            sigma_day = pm.HalfNormal("sigma_day", sigma=1.0)
            u_raw = pm.Normal("u_raw", 0, 1, dims="date")
            logit_p = logit_p + (sigma_day * u_raw)[date_idx_]

        # p 는 Deterministic 으로 저장하지 않는다(관측 17,000 x 사후표본 = 메모리 과다).
        pm.Bernoulli("obs", logit_p=logit_p, observed=y_, dims="obs_id")

    return model


# ----------------------------------------------------------------------
# 3. 주변예측확률 (u[t]를 사전분포에서 적분)
# ----------------------------------------------------------------------
def predict_marginal(idata, X, station_idx, day_pos, n_days,
                     n_draws=N_PRED_DRAWS, seed=RANDOM_SEED):
    """
    학습·테스트 모두 동일한 방식으로 계산:
      p = E_{posterior, u ~ N(0, sigma_day)} [ sigmoid(alpha_i + X·beta_i + u_t) ]
    미래 날짜의 공통 충격 u_t 는 관측되지 않으므로 사전분포에서 적분한다.
    """
    rng = np.random.default_rng(seed)
    post = idata.posterior
    # arviz>=1.0 / pymc>=6 은 posterior 가 DataTree 라 Dataset 으로 변환이 필요하다.
    post = post.to_dataset() if hasattr(post, "to_dataset") else post
    stacked = post.stack(sample=("chain", "draw"))
    S = stacked.sizes["sample"]
    sel = rng.choice(S, size=min(n_draws, S), replace=False)

    alpha = stacked["alpha_station"].values[:, sel]        # (station, S')
    beta = stacked["beta_station"].values[:, :, sel]       # (station, feature, S')
    has_u = "sigma_day" in stacked
    sigma_day = stacked["sigma_day"].values[sel] if has_u else None

    p_sum = np.zeros(len(X))
    for k in range(len(sel)):
        eta = alpha[station_idx, k] + (X * beta[station_idx, :, k]).sum(axis=1)
        if has_u:
            u = rng.normal(0.0, sigma_day[k], size=n_days)
            eta = eta + u[day_pos]
        p_sum += 1.0 / (1.0 + np.exp(-eta))
    return p_sum / len(sel)


def calibration_plot(y_true, p, title, path, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    cal_pred, cal_obs, cal_n = [], [], []
    for i in range(n_bins):
        m = b == i
        if m.sum() > 0:
            cal_pred.append(p[m].mean())
            cal_obs.append(y_true[m].mean())
            cal_n.append(m.sum())
    auc = roc_auc_score(y_true, p)
    brier = brier_score_loss(y_true, p)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="완벽한 보정선")
    plt.scatter(cal_pred, cal_obs, s=[n / 2 for n in cal_n], alpha=0.7)
    plt.xlabel("모델 예측확률 (bin 평균)")
    plt.ylabel("실제 이벤트 비율")
    plt.title(f"{title}\n(AUC={auc:.3f}, Brier={brier:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return auc, brier


# ----------------------------------------------------------------------
# 4. 실행
# ----------------------------------------------------------------------
def main():
    print("[1/7] 데이터 로드 및 시계열 분할...")
    daily = load_daily_events(INPUT_PATH)
    train, test, cutoff = time_split(daily)
    train, test, _ = standardize(train, test)
    stations, groups, station_idx, group_of_station, dates, date_idx = build_indices(train)

    print(f"  -> 전체 {len(daily)}행 / 학습 {len(train)}행(~{cutoff.date()}) / "
          f"테스트 {len(test)}행({(cutoff + pd.Timedelta(days=1)).date()}~{test['date'].max().date()})")
    print(f"  -> 이벤트 비율: 학습 {train['event'].mean():.1%}, 테스트 {test['event'].mean():.1%}")

    model = build_model(train, stations, groups, station_idx, group_of_station, dates, date_idx)

    print("[2/7] 사전예측검정...")
    with model:
        prior = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)
    prior_rate = prior.prior_predictive["obs"].values.mean(axis=-1).flatten()
    plt.figure(figsize=(7, 4.5))
    plt.hist(prior_rate, bins=40, density=True, alpha=0.6, label="사전예측 이벤트율")
    plt.axvline(train["event"].mean(), color="red", linestyle="--",
                label=f"관측 이벤트율 {train['event'].mean():.1%}")
    plt.xlabel("이벤트 발생률")
    plt.title("사전예측검정 (Stage 3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "haze_prior_predictive.png"), dpi=150)
    plt.close()

    print("[3/7] 계층 로지스틱 모델 샘플링 (학습기간만 사용)...")
    with model:
        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True,
        )

    print("[4/7] 수렴 진단...")
    diag_vars = ["gamma_group", "tau_station", "beta_global", "tau_beta"]
    if USE_DAY_EFFECT:
        diag_vars.append("sigma_day")
    print(az.summary(idata, var_names=diag_vars))
    rhat_ds = az.rhat(idata, var_names=diag_vars)
    ess_ds = az.ess(idata, var_names=diag_vars)
    diag = pd.DataFrame([{"variable": v,
                          "rhat_max": float(rhat_ds[v].max()),
                          "ess_min": float(ess_ds[v].min())} for v in rhat_ds.data_vars])
    diag.to_csv(os.path.join(OUTPUT_DIR, "haze_convergence.csv"), index=False)
    print(f"  r_hat 최대 {diag['rhat_max'].max():.4f} / ESS 최소 {diag['ess_min'].min():.0f}")

    az.plot_trace(idata, var_names=diag_vars)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "haze_trace_diagnostics.png"), dpi=150)
    plt.close()

    print("[5/7] forest plot...")
    az.plot_forest(idata, var_names=["alpha_station", "gamma_group"], combined=True)
    plt.title("Station별 haze 기저위험 및 그룹 고정효과 (로짓, 94% HDI)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "haze_forest_alpha.png"), dpi=150)
    plt.close()

    az.plot_forest(idata, var_names=["beta_global"], combined=True)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("예측변수 효과 (전체 평균, 94% HDI)\n일별 공통효과 반영 후")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "haze_forest_beta.png"), dpi=150)
    plt.close()

    # 오즈비 표 저장
    beta_s = idata.posterior["beta_global"].stack(sample=("chain", "draw")).values  # (feature, S)
    or_tbl = pd.DataFrame({
        "feature": FEATURES,
        "beta_mean": beta_s.mean(axis=1),
        "beta_hdi_3%": np.percentile(beta_s, 3, axis=1),
        "beta_hdi_97%": np.percentile(beta_s, 97, axis=1),
        "odds_ratio": np.exp(beta_s.mean(axis=1)),
        "OR_hdi_3%": np.exp(np.percentile(beta_s, 3, axis=1)),
        "OR_hdi_97%": np.exp(np.percentile(beta_s, 97, axis=1)),
    })
    or_tbl.to_csv(os.path.join(OUTPUT_DIR, "haze_odds_ratio.csv"), index=False)
    print(or_tbl.round(3))

    print("[6/7] 성능 평가 (학습기간 vs 홀드아웃 1년)...")
    st_map = {s: i for i, s in enumerate(stations)}

    X_tr = train[FEATURES].values
    p_tr = predict_marginal(idata, X_tr, station_idx, date_idx, len(dates))
    auc_tr, brier_tr = calibration_plot(
        train["event"].values, p_tr, "학습기간 (주변예측확률)",
        os.path.join(OUTPUT_DIR, "haze_calibration_train.png"))

    test = test[test["station"].isin(st_map)].reset_index(drop=True)
    X_te = test[FEATURES].values
    st_te = test["station"].map(st_map).values
    dates_te = np.sort(test["date"].unique())
    day_te = pd.DatetimeIndex(dates_te).get_indexer(pd.DatetimeIndex(test["date"]))
    p_te = predict_marginal(idata, X_te, st_te, day_te, len(dates_te))
    auc_te, brier_te = calibration_plot(
        test["event"].values, p_te, f"홀드아웃 {TEST_MONTHS}개월 (주변예측확률)",
        os.path.join(OUTPUT_DIR, "haze_calibration_test.png"))

    base_tr, base_te = train["event"].mean(), test["event"].mean()
    print(f"  학습:  AUC={auc_tr:.3f}, Brier={brier_tr:.3f} (base rate {base_tr:.1%})")
    print(f"  테스트: AUC={auc_te:.3f}, Brier={brier_te:.3f} (base rate {base_te:.1%})")

    with open(os.path.join(OUTPUT_DIR, "haze_model_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("[C4] 시계열 홀드아웃 평가 (랜덤 분할 아님)\n")
        f.write(f"학습기간: ~ {cutoff.date()} ({len(train)}행, base rate {base_tr:.4f})\n")
        f.write(f"테스트기간: {(cutoff + pd.Timedelta(days=1)).date()} ~ "
                f"{test['date'].max().date()} ({len(test)}행, base rate {base_te:.4f})\n\n")
        f.write(f"학습  AUC={auc_tr:.4f}  Brier={brier_tr:.4f}\n")
        f.write(f"테스트 AUC={auc_te:.4f}  Brier={brier_te:.4f}\n\n")
        f.write("두 값 모두 날짜별 공통효과 u[t]를 사전분포에서 적분한 주변예측확률 기준이며,\n"
                "따라서 학습기간 성능도 v1의 in-sample 적합값보다 보수적으로 산출된다.\n")

    print("[7/7] 결과 저장...")
    az.summary(idata, var_names=["alpha_station", "gamma_group", "beta_station", "beta_global"]).to_csv(
        os.path.join(OUTPUT_DIR, "haze_event_summary.csv")
    )
    print(f"완료. 결과는 {OUTPUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
