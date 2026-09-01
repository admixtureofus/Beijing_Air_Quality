"""
04_bayesian_changepoint.py
============================
베이지안 변화점 탐지 (Changepoint Detection) - Soft Sigmoid Switch 모델

목적
----
SARIMA 팀이 "추세가 있다/없다"를 본다면, 베이지안 파트는 "언제, 얼마나
급격하게 레벨이 바뀌었는지"를 불확실성(신뢰구간)과 함께 제시한다.
2013~2017년은 베이징 대기오염 방지 행동계획 등 정책 개입이 있었던 시기라,
이 시점 전후로 PM2.5 레벨이 실제로 이동했는지, 그리고 그 이동이
"도시 전체에서 동시에" 일어났는지 "station마다 제각각"이었는지를 계층모델로
같이 추정한다.

방법론 선택 이유 (discrete switchpoint 대신 soft-sigmoid를 쓴 이유)
------------------------------------------------------------
전통적인 pm.math.switch + DiscreteUniform(tau) 방식은 tau가 이산 변수라서
NUTS를 못 쓰고 Metropolis로 떨어져 수렴이 느리고 station마다/계층으로
공유하기도 번거롭다. 대신 시그모이드로 "부드러운 전환"을 모델링하면 tau가
연속변수가 되어 NUTS로 전체를 한 번에 효율적으로 샘플링할 수 있고,
station 간 tau를 계층적으로 공유하는 것도 자연스럽다.

  level(t) = level_pre + delta * sigmoid((t - tau) / s)

  - level_pre : 변화 이전 레벨
  - delta     : 변화 이후 레벨 이동량 (음수면 감소, 즉 정책 효과로 개선)
  - tau       : 변화가 일어난 시점 (월 단위 인덱스, station마다 다를 수 있음)
  - s         : 전환이 얼마나 급격한지 (작을수록 계단식에 가까움)

  tau_station ~ Normal(tau_global, tau_spread)
  -> tau_spread가 작으면 "도시 전체가 비슷한 시점에 바뀜"
  -> tau_spread가 크면 "station마다 변화 시점이 다름"

입력
----
01_preprocess.py 로 만든 전처리 완료 파일 (병합 + 시간보간 + datetime 포함).
아래 INPUT_PATH 를 본인 환경에 맞게 수정.

출력 (OUTPUT_DIR 아래)
----------------------
- changepoint_summary.csv        : station별 tau(변화시점), delta(변화량) 사후 요약
- changepoint_trace_diagnostics.png
- changepoint_forest_tau.png      : station별 변화 시점(실제 날짜로 환산) forest plot
- changepoint_forest_delta.png    : station별 변화량 forest plot
- changepoint_fit_examples.png    : 대표 station 2~3곳의 원자료 + 적합곡선
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 로컬 기준. Mac: AppleGothic
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------
INPUT_PATH = r"C:\Users\과표사업단\Documents\beijing\cleaned_prsa_all.csv"   
OUTPUT_DIR = r"C:\Users\과표사업단\Documents\beijing\04_bayesian_changepoint\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRAWS = 1500
TUNE = 1500
CHAINS = 2
TARGET_ACCEPT = 0.95   # 시그모이드 changepoint 모델은 posterior 지형이 까다로워 0.9~0.95 권장
RANDOM_SEED = 42


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
        .sort_values(["station", "month_period"])
    )
    monthly["log_pm25"] = np.log1p(monthly["PM2.5"])

    # 전체 station 공통 시간축 (0, 1, 2, ... T-1) 부여
    all_periods = sorted(monthly["month_period"].unique())
    period_to_idx = {p: i for i, p in enumerate(all_periods)}
    monthly["t_idx"] = monthly["month_period"].map(period_to_idx)
    monthly["month_of_year"] = monthly["month_period"].dt.month - 1  # 0~11, 계절성 통제용

    return monthly, all_periods


# ----------------------------------------------------------------------
# 2. 모델 정의
# ----------------------------------------------------------------------
def build_changepoint_model(monthly, stations):
    n_station = len(stations)
    T = monthly["t_idx"].max() + 1

    station_idx = monthly["station"].map({s: i for i, s in enumerate(stations)}).values
    t_idx = monthly["t_idx"].values
    month_idx = monthly["month_of_year"].values
    y = monthly["log_pm25"].values

    coords = {
        "station": stations,
        "month_of_year": list(range(12)),
        "obs_id": monthly.index,
    }

    with pm.Model(coords=coords) as model:
        station_idx_ = pm.Data("station_idx", station_idx, dims="obs_id")
        t_idx_ = pm.Data("t_idx", t_idx.astype(float), dims="obs_id")
        month_idx_ = pm.Data("month_idx", month_idx, dims="obs_id")
        y_ = pm.Data("y_obs", y, dims="obs_id")

        # -- 계절성 통제 (전 station 공통, 월별 더미) --
        # 첫 달(1월)을 기준으로 나머지 11개월 효과를 추정 (식별성 확보)
        month_effect_raw = pm.Normal("month_effect_raw", 0, 1, dims="month_of_year")
        month_effect = pm.Deterministic(
            "month_effect", month_effect_raw - month_effect_raw.mean(), dims="month_of_year"
        )

        # -- 변화 이전 레벨 (station별, 계층적) --
        mu_pre_global = pm.Normal("mu_pre_global", 4.0, 1.0)
        tau_pre = pm.HalfNormal("tau_pre", 1.0)
        pre_offset = pm.Normal("pre_offset", 0, 1, dims="station")
        level_pre = pm.Deterministic(
            "level_pre", mu_pre_global + tau_pre * pre_offset, dims="station"
        )

        # -- 변화량 delta (station별, 계층적: 정책 효과가 균일한지 아닌지 확인) --
        mu_delta_global = pm.Normal("mu_delta_global", 0, 1.0)
        tau_delta = pm.HalfNormal("tau_delta", 1.0)
        delta_offset = pm.Normal("delta_offset", 0, 1, dims="station")
        delta = pm.Deterministic(
            "delta", mu_delta_global + tau_delta * delta_offset, dims="station"
        )

        # -- 변화 시점 tau (station별, 계층적: 도시 전체 동시 vs station별 상이) --
        tau_global = pm.Normal("tau_global", T / 2, T / 4)
        tau_spread = pm.HalfNormal("tau_spread", T / 6)
        tau_offset = pm.Normal("tau_offset", 0, 1, dims="station")
        tau_station = pm.Deterministic(
            "tau_station", tau_global + tau_spread * tau_offset, dims="station"
        )

        # -- 전환 급격도 (station 공통) --
        s_sharpness = pm.HalfNormal("s_sharpness", 3.0)

        # -- 결합 --
        switch = pm.math.sigmoid((t_idx_ - tau_station[station_idx_]) / s_sharpness)
        mu = (
            level_pre[station_idx_]
            + delta[station_idx_] * switch
            + month_effect[month_idx_]
        )
        sigma_obs = pm.HalfNormal("sigma_obs", 1.0)

        pm.Normal("obs", mu=mu, sigma=sigma_obs, observed=y_, dims="obs_id")

    return model


# ----------------------------------------------------------------------
# 3. 실행
# ----------------------------------------------------------------------
def main():
    print("[1/5] 데이터 로드 및 월단위 집계...")
    monthly, all_periods = load_monthly(INPUT_PATH)
    stations = sorted(monthly["station"].unique())
    print(f"  -> {len(monthly)} station-month rows, {len(stations)} stations, "
          f"{len(all_periods)}개월 ({all_periods[0]} ~ {all_periods[-1]})")

    print("[2/5] 변화점 모델 샘플링 (시그모이드 soft-changepoint)...")
    model = build_changepoint_model(monthly, stations)
    with model:
        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True,
        )

    print("[3/5] 수렴 진단...")
    summary = az.summary(idata, var_names=[
        "tau_global", "tau_spread", "mu_delta_global", "tau_delta",
        "mu_pre_global", "tau_pre", "s_sharpness", "sigma_obs",
    ])
    print(summary)
    max_rhat = summary["r_hat"].max()
    if pd.notna(max_rhat) and max_rhat > 1.01:
        print(f"  [경고] r_hat 최대값 {max_rhat:.3f} > 1.01 -> tune/draws를 늘려 재실행 권장")
    elif pd.isna(max_rhat):
        print("  [주의] r_hat이 계산되지 않음 -> 로컬 arviz 버전에서 az.rhat(idata)로 별도 확인 권장")
    else:
        print(f"  수렴 양호: r_hat 최대값 {max_rhat:.3f}")

    az.plot_trace(idata, var_names=["tau_global", "tau_spread", "mu_delta_global", "s_sharpness"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "changepoint_trace_diagnostics.png"), dpi=150)
    plt.close()

    # -- tau_global을 실제 날짜로 환산해서 출력 --
    tau_global_samples = idata.posterior["tau_global"].values.flatten()
    tau_mean_month = int(round(tau_global_samples.mean()))
    tau_mean_month = min(max(tau_mean_month, 0), len(all_periods) - 1)
    print(f"  -> 도시 전체 평균 변화 시점(tau_global) 추정: "
          f"약 {all_periods[tau_mean_month]} 무렵 (사후평균 t={tau_global_samples.mean():.1f})")

    print("[4/5] station별 결과 저장 (forest plot)...")
    # tau: 월 인덱스를 실제 라벨로 바꿔서 forest plot이 해석 가능하도록 준비
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

    # -- 대표 station 2~3곳 원자료 + 적합곡선 시각화 --
    print("[5/5] 적합 예시 시각화 및 요약 저장...")
    example_stations = stations[:3]
    fig, axes = plt.subplots(len(example_stations), 1, figsize=(9, 3 * len(example_stations)), sharex=True)
    if len(example_stations) == 1:
        axes = [axes]

    level_pre_mean = idata.posterior["level_pre"].mean(dim=["chain", "draw"]).values
    delta_mean = idata.posterior["delta"].mean(dim=["chain", "draw"]).values
    tau_mean = idata.posterior["tau_station"].mean(dim=["chain", "draw"]).values
    s_mean = idata.posterior["s_sharpness"].mean().item()
    month_effect_mean = idata.posterior["month_effect"].mean(dim=["chain", "draw"]).values

    t_grid = np.arange(len(all_periods))
    for ax, st in zip(axes, example_stations):
        i = stations.index(st)
        sub = monthly[monthly["station"] == st].sort_values("t_idx")
        ax.scatter(sub["t_idx"], sub["log_pm25"], s=15, color="black", label="관측값 (월평균)")

        # 적합곡선: 계절성 평균 효과를 제외한 "레벨" 궤적만 표시 (changepoint 해석용)
        switch = 1 / (1 + np.exp(-(t_grid - tau_mean[i]) / s_mean))
        fitted_level = level_pre_mean[i] + delta_mean[i] * switch
        ax.plot(t_grid, fitted_level, color="tab:blue", linewidth=2, label="추정된 레벨 궤적")
        ax.axvline(tau_mean[i], color="tab:red", linestyle="--", linewidth=1,
                   label=f"추정 변화시점 (t={tau_mean[i]:.1f})")
        ax.set_title(st)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("월 인덱스 (0 = " + str(all_periods[0]) + ")")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "changepoint_fit_examples.png"), dpi=150)
    plt.close()

    full_summary = az.summary(idata, var_names=["tau_station", "delta", "level_pre"])
    full_summary.to_csv(os.path.join(OUTPUT_DIR, "changepoint_summary.csv"))

    print(f"완료. 결과는 {OUTPUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
