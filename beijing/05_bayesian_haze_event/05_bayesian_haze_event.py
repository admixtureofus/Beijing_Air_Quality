"""
05_bayesian_haze_event.py
============================
베이지안 계층 로지스틱 회귀 - 고농도(haze) 이벤트 확률 모델

목적
----
팀 공통 주제인 "고농도 스모그 이벤트 예측"에 베이지안 방식으로 직접 기여한다.
XGBoost 팀이 point prediction(예/아니오, 혹은 확률 하나)을 낸다면, 이 모델은
station별 기저위험(baseline risk)과 각 예측변수 효과에 대한 **불확실성**을
사후분포(HDI)로 함께 제공하는 것이 차별점이다.

이벤트 정의
----------
중국 AQI 기준 "중도오염 이상"에 해당하는 일평균 PM2.5 > 150 ug/m3 를
haze event(=1)로 정의한다. station별 일평균 데이터에서 이벤트 비율은
약 9~15% 로 station마다 다르다 (Dingling·Huairou 낮음, 도심권 높음) ->
station 정보 공유가 여기서도 의미 있다.

모델
----
logit(p_it) = alpha_station[i] + beta_station[i] · X_it

  - alpha_station : station별 기저위험 (그룹별 부분 풀링, Stage1과 동일한
    지리적 3그룹 구조 재사용 -> 일관된 스토리)
  - beta_station  : 기상변수 + 전날 농도(lag) 효과 (station별 부분 풀링)
  - lag_logpm25   : 전날 log(PM2.5) 평균 -> 스모그의 지속성(persistence)을
    반영. 이 항이 빠지면 "어제도 나빴으면 오늘도 나쁠 확률이 높다"는
    가장 강력한 정보를 모델이 놓치게 된다.
  - sin_doy/cos_doy : 연중 계절성을 2개 변수로 압축 (cyclic encoding).
    월별 더미(12개) 대신 사용하면 파라미터 수가 줄고 겨울-여름 경계가
    부드럽게 연결된다.

입력
----
01_preprocess.py 로 만든 전처리 완료 파일 (병합 + 시간보간 + datetime 포함).
아래 INPUT_PATH 를 본인 환경에 맞게 수정.

출력 (OUTPUT_DIR 아래)
----------------------
- haze_event_summary.csv         : station별 alpha, beta 사후 요약
- haze_trace_diagnostics.png
- haze_forest_alpha.png          : station별 기저위험(로짓 스케일) forest plot
- haze_forest_beta.png           : 예측변수별 효과(station 평균) forest plot
- haze_calibration.png           : 예측확률 vs 실제 이벤트 비율 (calibration curve)
- haze_model_metrics.txt         : AUC, Brier score 등 예측 성능 요약
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 로컬 기준. Mac: AppleGothic
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------
INPUT_PATH = r"C:\Users\과표사업단\Documents\beijing\cleaned_prsa_all.csv"   # 지홍 환경: 01_preprocess.py 출력 파일 경로로 교체
OUTPUT_DIR = r"/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENT_THRESHOLD = 150.0  # 일평균 PM2.5 (ug/m3) 기준

DRAWS = 1500
TUNE = 1500
CHAINS = 2
TARGET_ACCEPT = 0.9
RANDOM_SEED = 42

STATION_GROUP = {
    "Aotizhongxin": "urban", "Dongsi": "urban", "Guanyuan": "urban",
    "Wanliu": "urban", "Wanshouxigong": "urban", "Nongzhanguan": "urban",
    "Tiantan": "urban",
    "Gucheng": "suburban_industrial", "Shunyi": "suburban_industrial",
    "Changping": "suburban_industrial",
    "Dingling": "background", "Huairou": "background",
}

FEATURES = ["TEMP_z", "PRES_z", "DEWP_z", "WSPM_z", "RAIN_z", "lag_logpm25_z", "sin_doy", "cos_doy"]


# ----------------------------------------------------------------------
# 1. 데이터 준비: station별 일단위 이벤트 + lag + 계절 인코딩
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

    # station별로 정렬 후 lag(전날 log_pm25) 생성 -> 첫날은 NaN이라 드롭됨
    daily = daily.sort_values(["station", "date"])
    daily["lag_logpm25"] = daily.groupby("station")["log_pm25"].shift(1)

    # 계절 cyclic encoding
    doy = daily["date"].dt.dayofyear
    daily["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    daily["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    daily = daily.dropna(subset=["event", "TEMP", "PRES", "DEWP", "WSPM", "RAIN", "lag_logpm25"])

    # 표준화 (lag_logpm25 포함, RAIN은 대부분 0이라 표준화해도 분포 치우침 -> 그대로 두되 스케일만 정규화)
    for raw, z in [("TEMP", "TEMP_z"), ("PRES", "PRES_z"), ("DEWP", "DEWP_z"),
                   ("WSPM", "WSPM_z"), ("RAIN", "RAIN_z"), ("lag_logpm25", "lag_logpm25_z")]:
        daily[z] = (daily[raw] - daily[raw].mean()) / daily[raw].std()

    return daily


def build_indices(daily):
    stations = sorted(daily["station"].unique())
    groups = sorted(daily["group"].unique())
    station_idx = daily["station"].map({s: i for i, s in enumerate(stations)}).values
    group_of_station = np.array([groups.index(STATION_GROUP[s]) for s in stations])
    return stations, groups, station_idx, group_of_station


# ----------------------------------------------------------------------
# 2. 모델 정의: 계층 로지스틱 회귀 (station 절편 + 계수 부분 풀링)
# ----------------------------------------------------------------------
def build_model(daily, stations, groups, station_idx, group_of_station):
    X = daily[FEATURES].values
    y = daily["event"].values

    coords = {"station": stations, "group": groups, "feature": FEATURES, "obs_id": daily.index}

    with pm.Model(coords=coords) as model:
        station_idx_ = pm.Data("station_idx", station_idx, dims="obs_id")
        group_of_station_ = pm.Data("group_of_station", group_of_station, dims="station")
        X_ = pm.Data("X", X, dims=("obs_id", "feature"))
        y_ = pm.Data("y_obs", y, dims="obs_id")

        # -- 절편: station <- 그룹 <- 전체 (Stage1과 동일한 부분 풀링 구조) --
        mu_global = pm.Normal("mu_global", mu=-2.0, sigma=1.5)  # 사전 이벤트율 ~10% 근방에서 시작
        tau_group = pm.HalfNormal("tau_group", sigma=1.0)
        group_offset = pm.Normal("group_offset", 0, 1, dims="group")
        mu_group = pm.Deterministic("mu_group", mu_global + tau_group * group_offset, dims="group")

        tau_station = pm.HalfNormal("tau_station", sigma=1.0)
        station_offset = pm.Normal("station_offset", 0, 1, dims="station")
        alpha_station = pm.Deterministic(
            "alpha_station", mu_group[group_of_station_] + tau_station * station_offset, dims="station"
        )

        # -- 계수: station별 부분 풀링 --
        beta_global = pm.Normal("beta_global", 0, 1, dims="feature")
        tau_beta = pm.HalfNormal("tau_beta", sigma=0.5, dims="feature")
        beta_offset = pm.Normal("beta_offset", 0, 1, dims=("station", "feature"))
        beta_station = pm.Deterministic(
            "beta_station", beta_global + tau_beta * beta_offset, dims=("station", "feature")
        )

        logit_p = alpha_station[station_idx_] + (X_ * beta_station[station_idx_]).sum(axis=-1)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p), dims="obs_id")

        pm.Bernoulli("obs", p=p, observed=y_, dims="obs_id")

    return model


# ----------------------------------------------------------------------
# 3. 실행
# ----------------------------------------------------------------------
def main():
    print("[1/6] 데이터 로드, 일단위 이벤트 정의, lag/계절 변수 생성...")
    daily = load_daily_events(INPUT_PATH)
    stations, groups, station_idx, group_of_station = build_indices(daily)
    print(f"  -> {len(daily)} station-day rows, 전체 이벤트 비율 {daily['event'].mean():.1%}")
    print(daily.groupby("station")["event"].mean().sort_values())

    print("[2/6] 계층 로지스틱 모델 샘플링...")
    model = build_model(daily, stations, groups, station_idx, group_of_station)
    with model:
        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True,
        )

    print("[3/6] 수렴 진단...")
    summary = az.summary(idata, var_names=["mu_global", "tau_group", "tau_station",
                                            "beta_global", "tau_beta"])
    print(summary)
    max_rhat = summary["r_hat"].max()
    if pd.notna(max_rhat) and max_rhat > 1.01:
        print(f"  [경고] r_hat 최대값 {max_rhat:.3f} > 1.01 -> tune/draws를 늘려 재실행 권장")
    elif pd.isna(max_rhat):
        print("  [주의] r_hat이 계산되지 않음 -> 로컬 arviz 버전에서 az.rhat(idata)로 별도 확인 권장")
    else:
        print(f"  수렴 양호: r_hat 최대값 {max_rhat:.3f}")

    az.plot_trace(idata, var_names=["mu_global", "tau_group", "tau_station", "beta_global"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "haze_trace_diagnostics.png"), dpi=150)
    plt.close()

    print("[4/6] station별 기저위험 / 계수 forest plot...")
    az.plot_forest(idata, var_names=["alpha_station"], combined=True)
    plt.title("Station별 haze 이벤트 기저위험 (로짓 스케일, 94% HDI)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "haze_forest_alpha.png"), dpi=150)
    plt.close()

    az.plot_forest(idata, var_names=["beta_global"], combined=True)
    plt.title("예측변수 효과 (전체 평균, 94% HDI)\n(0을 넘지 않으면 뚜렷한 효과)")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "haze_forest_beta.png"), dpi=150)
    plt.close()

    print("[5/6] 예측 성능 평가 (AUC, calibration)...")
    p_mean = idata.posterior["p"].mean(dim=["chain", "draw"]).values
    y_true = daily["event"].values

    auc = roc_auc_score(y_true, p_mean)
    brier = brier_score_loss(y_true, p_mean)

    # calibration curve (10-bin)
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(p_mean, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)
    cal_pred, cal_obs, cal_n = [], [], []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() > 0:
            cal_pred.append(p_mean[mask].mean())
            cal_obs.append(y_true[mask].mean())
            cal_n.append(mask.sum())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="완벽한 보정선")
    plt.scatter(cal_pred, cal_obs, s=[n / 2 for n in cal_n], alpha=0.7)
    plt.xlabel("모델 예측확률 (bin 평균)")
    plt.ylabel("실제 이벤트 비율")
    plt.title(f"Calibration Curve (AUC={auc:.3f}, Brier={brier:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "haze_calibration.png"), dpi=150)
    plt.close()

    with open(os.path.join(OUTPUT_DIR, "haze_model_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Brier score: {brier:.4f}\n")
        f.write(f"이벤트 비율(base rate): {y_true.mean():.4f}\n")
        f.write("\n주의: 이 지표는 in-sample (학습에 쓴 데이터로 평가) 기준입니다.\n"
                "XGBoost 팀의 held-out 성능과 직접 비교하려면 train/test 분할이 필요합니다.\n")
    print(f"  AUC={auc:.3f}, Brier={brier:.3f}")

    print("[6/6] 결과 저장...")
    full_summary = az.summary(idata, var_names=["alpha_station", "mu_group", "beta_station"])
    full_summary.to_csv(os.path.join(OUTPUT_DIR, "haze_event_summary.csv"))

    print(f"완료. 결과는 {OUTPUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
