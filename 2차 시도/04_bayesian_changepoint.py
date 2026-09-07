"""
04_bayesian_changepoint.py  (v2)
=========================================================
베이지안 변화점 탐지 (Changepoint Detection) - Soft Sigmoid Switch 모델

v1 대비 변경점 (교수님 코멘트 반영)
-----------------------------------
[C0] 모든 파라미터의 사전분포를 아래 PRIOR SPECIFICATION 에 명시하고,
     사전예측검정(prior predictive check) 그림을 산출.

[C2] 잔차 AR(1) 구조 추가
     월평균으로 집계해도 인접 월의 오염 수준은 서로 강하게 연결되어 있다.
     station별 월 시계열에 조건부 AR(1) 오차를 도입하여, 변화시점(tau)과
     변화량(delta)의 신용구간이 과소추정되지 않도록 했다.
       y[i,t] ~ Normal( mu[i,t] + rho*(y[i,t-1] - mu[i,t-1]), sigma_e )
     시계열 시작 시점은 정상분포 주변분산 sigma_e/sqrt(1-rho^2) 사용.

[C3] 반복측정(같은 도시 12 station) 구조 추가
     같은 달에는 12개 station이 사실상 동일한 기상·배출 조건을 공유하므로,
     월별 도시 공통 랜덤효과 v[t] ~ Normal(0, sigma_time) 를 추가했다.
       - v[t]가 없으면 12개 station이 독립 반복인 것처럼 취급되어
         tau_global의 구간이 실제보다 12배가량 좁게 나온다.
       - sigma_time 에는 약간 강한 사전분포 HalfNormal(0.2)를 두었다.
         v[t]가 지나치게 유연해지면 시그모이드 레벨 이동 자체를 흡수해버려
         변화점이 식별되지 않기 때문이다(식별성 관리). 계절성(month_effect)은
         12개월 주기로 반복되고 v[t]는 반복되지 않으므로 둘은 구분 가능하다.

[C1] 관련: 본 모델에는 3개 지역 그룹 계층이 없고, 계층은 12개 station 수준에만 존재한다.
     12개 단위는 분산모수(tau_*)를 추정하기에 충분하므로 랜덤효과를 유지했다.

모델
----
  level_i(t) = level_pre_i + delta_i * sigmoid((t - tau_i) / s)
  mu_i(t)    = level_i(t) + month_effect[month_of_year(t)] + v[t]

PRIOR SPECIFICATION (C0)
------------------------
  mu_pre_global    ~ Normal(4.0, 1.0)      : 변화 전 도시 평균 log1p(PM2.5) 수준.
  tau_pre          ~ HalfNormal(0.5)       : 변화 전 레벨의 station 간 편차.
  mu_delta_global  ~ Normal(0, 0.5)        : 평균 변화량. 0을 중심에 두어 "감소했다"는
                                             결론이 사전분포가 아니라 데이터에서 나오게 함.
  tau_delta        ~ HalfNormal(0.5)       : 변화량의 station 간 이질성.
  tau_global       ~ Normal(T/2, T/4)      : 변화 시점의 도시 평균. 관측구간 전체에 걸친
                                             약정보 사전분포(특정 시점을 선호하지 않음).
  tau_spread       ~ HalfNormal(T/6)       : station 간 변화시점 산포.
  s_sharpness      ~ HalfNormal(3.0)       : 전환의 완만함(월 단위). 사후분포가 작으면 계단형.
  month_effect_raw ~ Normal(0, 1) (12개)   : 계절 고정효과, 평균 0으로 센터링해 식별성 확보.
  sigma_time       ~ HalfNormal(0.2)       : 월별 도시 공통 충격의 크기 (C3).
  v_raw[t]         ~ Normal(0, 1)          : 비중심화 보조변수.
  rho              ~ TruncatedNormal(0.4, 0.3, -0.95, 0.95) : 월간 AR(1) 계수 (C2).
  sigma_e          ~ HalfNormal(1.0)       : station-month 고유 잔차.
  *_offset         ~ Normal(0, 1)          : 모든 계층모수의 비중심화 보조변수.

출력 (OUTPUT_DIR 아래)
----------------------
- changepoint_prior_predictive.png
- changepoint_summary.csv
- changepoint_convergence.csv
- changepoint_trace_diagnostics.png
- changepoint_forest_tau.png / changepoint_forest_delta.png
- changepoint_fit_examples.png
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"   # Mac: AppleGothic
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------
INPUT_PATH = r"파일경로\cleaned_prsa_all.csv"
OUTPUT_DIR = r"저장위치\04_bayesian_changepoint\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRAWS = 2000
TUNE = 2000
CHAINS = 4
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42

USE_AR1 = True          # C2
USE_TIME_EFFECT = True  # C3
SIGMA_TIME_PRIOR = 0.2  # C3 식별성 관리를 위한 사전분포 스케일


# ----------------------------------------------------------------------
# 1. 데이터 준비: station별 월평균 log(PM2.5) 시계열
# ----------------------------------------------------------------------
def load_monthly(path):
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df = df.dropna(subset=["PM2.5"])

    df["month_period"] = df["datetime"].dt.to_period("M")
    monthly = (
        df.groupby(["station", "month_period"])["PM2.5"]
        .mean()
        .reset_index()
    )
    monthly["log_pm25"] = np.log1p(monthly["PM2.5"])

    all_periods = sorted(monthly["month_period"].unique())
    period_to_idx = {p: i for i, p in enumerate(all_periods)}
    monthly["t_idx"] = monthly["month_period"].map(period_to_idx)
    monthly["month_of_year"] = monthly["month_period"].dt.month - 1

    # AR(1) 계산을 위해 station -> 시간 순 정렬 후 인덱스 재부여
    monthly = monthly.sort_values(["station", "t_idx"]).reset_index(drop=True)
    return monthly, all_periods


def build_ar_index(monthly):
    """같은 station의 직전 '월' 행 인덱스와 존재 여부 플래그 (월이 끊기면 체인 단절)."""
    n = len(monthly)
    prev_idx = np.arange(n)
    has_prev = np.zeros(n)
    for _, g in monthly.groupby("station"):
        idx = g.index.values
        gap = g["t_idx"].diff().values
        for j in range(1, len(idx)):
            if gap[j] == 1:
                prev_idx[idx[j]] = idx[j - 1]
                has_prev[idx[j]] = 1.0
    return prev_idx, has_prev


# ----------------------------------------------------------------------
# 2. 모델 정의
# ----------------------------------------------------------------------
def build_changepoint_model(monthly, stations, all_periods, prev_idx, has_prev):
    T = int(monthly["t_idx"].max()) + 1
    station_idx = monthly["station"].map({s: i for i, s in enumerate(stations)}).values
    t_idx = monthly["t_idx"].values
    month_idx = monthly["month_of_year"].values
    y = monthly["log_pm25"].values

    coords = {
        "station": stations,
        "month_of_year": list(range(12)),
        "time": [str(p) for p in all_periods],
        "obs_id": np.arange(len(monthly)),
    }

    with pm.Model(coords=coords) as model:
        station_idx_ = pm.Data("station_idx", station_idx, dims="obs_id")
        t_idx_ = pm.Data("t_idx", t_idx.astype(float), dims="obs_id")
        t_pos_ = pm.Data("t_pos", t_idx, dims="obs_id")          # 정수 인덱싱용
        month_idx_ = pm.Data("month_idx", month_idx, dims="obs_id")
        prev_idx_ = pm.Data("prev_idx", prev_idx, dims="obs_id")
        has_prev_ = pm.Data("has_prev", has_prev, dims="obs_id")
        y_prev_ = pm.Data("y_prev", y[prev_idx], dims="obs_id")
        y_ = pm.Data("y_obs", y, dims="obs_id")

        # -- 계절 고정효과 (평균 0 제약으로 식별) --
        month_effect_raw = pm.Normal("month_effect_raw", 0, 1, dims="month_of_year")
        month_effect = pm.Deterministic(
            "month_effect", month_effect_raw - month_effect_raw.mean(), dims="month_of_year"
        )

        # -- 변화 이전 레벨 (station별 부분 풀링) --
        mu_pre_global = pm.Normal("mu_pre_global", 4.0, 1.0)
        tau_pre = pm.HalfNormal("tau_pre", 0.5)
        pre_offset = pm.Normal("pre_offset", 0, 1, dims="station")
        level_pre = pm.Deterministic(
            "level_pre", mu_pre_global + tau_pre * pre_offset, dims="station"
        )

        # -- 변화량 delta --
        mu_delta_global = pm.Normal("mu_delta_global", 0, 0.5)
        tau_delta = pm.HalfNormal("tau_delta", 0.5)
        delta_offset = pm.Normal("delta_offset", 0, 1, dims="station")
        delta = pm.Deterministic(
            "delta", mu_delta_global + tau_delta * delta_offset, dims="station"
        )

        # -- 변화 시점 tau --
        tau_global = pm.Normal("tau_global", T / 2, T / 4)
        tau_spread = pm.HalfNormal("tau_spread", T / 6)
        tau_offset = pm.Normal("tau_offset", 0, 1, dims="station")
        tau_station = pm.Deterministic(
            "tau_station", tau_global + tau_spread * tau_offset, dims="station"
        )

        s_sharpness = pm.HalfNormal("s_sharpness", 3.0)

        switch = pm.math.sigmoid((t_idx_ - tau_station[station_idx_]) / s_sharpness)
        mu = (
            level_pre[station_idx_]
            + delta[station_idx_] * switch
            + month_effect[month_idx_]
        )

        # -- [C3] 월별 도시 공통 랜덤효과 --
        if USE_TIME_EFFECT:
            sigma_time = pm.HalfNormal("sigma_time", SIGMA_TIME_PRIOR)
            v_raw = pm.Normal("v_raw", 0, 1, dims="time")
            v = sigma_time * v_raw
            mu = mu + v[t_pos_]

        # -- [C2] 잔차 AR(1) --
        sigma_e = pm.HalfNormal("sigma_e", 1.0)
        if USE_AR1:
            rho = pm.TruncatedNormal("rho", mu=0.4, sigma=0.3, lower=-0.95, upper=0.95)
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
# 3. 실행
# ----------------------------------------------------------------------
def main():
    print("[1/6] 데이터 로드 및 월단위 집계...")
    monthly, all_periods = load_monthly(INPUT_PATH)
    stations = sorted(monthly["station"].unique())
    prev_idx, has_prev = build_ar_index(monthly)
    print(f"  -> {len(monthly)} station-month rows, {len(stations)} stations, "
          f"{len(all_periods)}개월 ({all_periods[0]} ~ {all_periods[-1]})")

    model = build_changepoint_model(monthly, stations, all_periods, prev_idx, has_prev)

    print("[2/6] 사전예측검정...")
    with model:
        prior = pm.sample_prior_predictive(draws=200, random_seed=RANDOM_SEED)
    prior_y = prior.prior_predictive["obs"].values.flatten()
    plt.figure(figsize=(7, 4.5))
    plt.hist(prior_y, bins=80, density=True, alpha=0.5, label="사전예측분포")
    plt.hist(monthly["log_pm25"], bins=40, density=True, alpha=0.5, label="관측값(월평균)")
    plt.xlabel("log1p(PM2.5)")
    plt.title("사전예측검정 (Stage 2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "changepoint_prior_predictive.png"), dpi=150)
    plt.close()

    print("[3/6] 변화점 모델 샘플링...")
    with model:
        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True,
        )

    print("[4/6] 수렴 진단...")
    diag_vars = ["tau_global", "tau_spread", "mu_delta_global", "tau_delta",
                 "mu_pre_global", "tau_pre", "s_sharpness", "sigma_e"]
    if USE_TIME_EFFECT:
        diag_vars.append("sigma_time")
    if USE_AR1:
        diag_vars.append("rho")

    print(az.summary(idata, var_names=diag_vars))
    rhat_ds = az.rhat(idata, var_names=diag_vars)
    ess_ds = az.ess(idata, var_names=diag_vars)
    diag = pd.DataFrame([{"variable": v,
                          "rhat_max": float(rhat_ds[v].max()),
                          "ess_min": float(ess_ds[v].min())} for v in rhat_ds.data_vars])
    diag.to_csv(os.path.join(OUTPUT_DIR, "changepoint_convergence.csv"), index=False)
    print(f"  r_hat 최대 {diag['rhat_max'].max():.4f} / ESS 최소 {diag['ess_min'].min():.0f}")

    az.plot_trace(idata, var_names=diag_vars)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "changepoint_trace_diagnostics.png"), dpi=150)
    plt.close()

    # -- tau_global 을 실제 날짜로 환산 --
    tau_s = idata.posterior["tau_global"].values.flatten()
    lo, hi = np.percentile(tau_s, [3, 97])
    def to_period(t):
        t = int(round(min(max(t, 0), len(all_periods) - 1)))
        return all_periods[t]
    print(f"  -> tau_global 사후평균 t={tau_s.mean():.1f} ({to_period(tau_s.mean())}), "
          f"94% HDI t={lo:.1f}~{hi:.1f} ({to_period(lo)} ~ {to_period(hi)})")

    if USE_AR1:
        rho_s = idata.posterior["rho"].values.flatten()
        print(f"  -> 월간 AR(1) rho 사후평균 {rho_s.mean():.3f}")
    if USE_TIME_EFFECT:
        st_s = idata.posterior["sigma_time"].values.flatten()
        se_s = idata.posterior["sigma_e"].values.flatten()
        print(f"  -> sigma_time {st_s.mean():.3f} vs sigma_e {se_s.mean():.3f} "
              f"(같은 달 12 station 공유 성분의 크기)")

    print("[5/6] forest plot 저장...")
    az.plot_forest(idata, var_names=["tau_station"], combined=True)
    plt.title("Station별 변화 시점(tau, month index) 사후분포 (94% HDI)\n"
              f"t=0 -> {all_periods[0]}, t={len(all_periods)-1} -> {all_periods[-1]}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "changepoint_forest_tau.png"), dpi=150)
    plt.close()

    az.plot_forest(idata, var_names=["delta"], combined=True)
    plt.title("Station별 변화량(delta, log-scale) 사후분포 (94% HDI)\n(음수 = 감소/개선)")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "changepoint_forest_delta.png"), dpi=150)
    plt.close()

    print("[6/6] 적합 예시 시각화 및 요약 저장...")
    example_stations = stations[:3]
    fig, axes = plt.subplots(len(example_stations), 1,
                             figsize=(9, 3 * len(example_stations)), sharex=True)
    if len(example_stations) == 1:
        axes = [axes]

    level_pre_mean = idata.posterior["level_pre"].mean(dim=["chain", "draw"]).values
    delta_mean = idata.posterior["delta"].mean(dim=["chain", "draw"]).values
    tau_mean = idata.posterior["tau_station"].mean(dim=["chain", "draw"]).values
    tau_post = idata.posterior["tau_station"].values.reshape(-1, len(stations))
    s_mean = float(idata.posterior["s_sharpness"].mean())

    t_grid = np.arange(len(all_periods))
    for ax, st in zip(axes, example_stations):
        i = stations.index(st)
        sub = monthly[monthly["station"] == st].sort_values("t_idx")
        ax.scatter(sub["t_idx"], sub["log_pm25"], s=15, color="black", label="관측값(월평균)")
        switch = 1 / (1 + np.exp(-(t_grid - tau_mean[i]) / s_mean))
        ax.plot(t_grid, level_pre_mean[i] + delta_mean[i] * switch,
                color="tab:blue", linewidth=2, label="추정된 레벨 궤적")
        tlo, thi = np.percentile(tau_post[:, i], [3, 97])
        ax.axvspan(tlo, thi, color="tab:red", alpha=0.15, label="변화시점 94% HDI")
        ax.axvline(tau_mean[i], color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(st)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel(f"월 인덱스 (0 = {all_periods[0]})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "changepoint_fit_examples.png"), dpi=150)
    plt.close()

    az.summary(idata, var_names=["tau_station", "delta", "level_pre"]).to_csv(
        os.path.join(OUTPUT_DIR, "changepoint_summary.csv")
    )
    print(f"완료. 결과는 {OUTPUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
