"""
03_bayesian_hierarchical.py
============================
베이지안 계층 회귀 모델 (Hierarchical / Partial Pooling Regression)

목적
----
station마다 독립적으로 회귀식을 적합하면(no pooling) 관측이 짧거나 변동성이
큰 station(예: Dingling, Huairou)에서 추정이 불안정해지고, 전체를 하나로
합쳐 적합하면(complete pooling) station 간 실제 차이를 무시하게 된다.
이 스크립트는 그 중간 지점인 부분 풀링(partial pooling)을 통해
"station 간 정보 공유" 를 구현한다.

  station_intercept ~ Normal(group_intercept, tau_station)
  group_intercept   ~ Normal(global_intercept, tau_group)

입력
----
01_preprocess.py 로 만든 전처리 완료 파일 (병합 + 시간보간 + datetime 포함)
아래 INPUT_PATH 를 본인 환경에 맞게 수정.

출력 (OUTPUT_DIR 아래)
----------------------
- model_comparison.csv      : pooled / unpooled / hierarchical LOO 비교
- posterior_summary.csv     : station별 파라미터 사후분포 요약 (평균, 94% HDI, r_hat)
- station_intercepts_forest.png : station별 절편 forest plot
- ppc_check.png             : posterior predictive check
- trace_diagnostics.png     : 수렴 진단 trace plot
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows 로컬 환경 기준. 다른 OS면 아래를 상황에 맞게 교체:
#  - Mac: "AppleGothic"  /  Linux(Colab 등): "NanumGothic" 설치 후 지정)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------
INPUT_PATH = r"불러올 전처리 파일 경로(파일명.확장자까지)"
OUTPUT_DIR = r"저장할 위치"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MCMC 설정: 로컬(코어 여러 개) 환경에서는 DRAWS/TUNE을 늘리고 CHAINS=4 권장
DRAWS = 1000
TUNE = 1000
CHAINS = 2
TARGET_ACCEPT = 0.9
RANDOM_SEED = 42

# station -> 지역 그룹 매핑 (station_pollutant_summary.csv 의 평균값 패턴 참고:
#  Dingling·Huairou 는 뚜렷이 낮고, 나머지 도심권은 서로 비슷하게 높음 -> 3그룹)
STATION_GROUP = {
    "Aotizhongxin": "urban", "Dongsi": "urban", "Guanyuan": "urban",
    "Wanliu": "urban", "Wanshouxigong": "urban", "Nongzhanguan": "urban",
    "Tiantan": "urban",
    "Gucheng": "suburban_industrial", "Shunyi": "suburban_industrial",
    "Changping": "suburban_industrial",
    "Dingling": "background", "Huairou": "background",
}

FEATURES = ["TEMP", "PRES", "DEWP", "WSPM"]  # 필요시 RAIN, season 더미 등 추가 가능


# ----------------------------------------------------------------------
# 1. 데이터 준비
# ----------------------------------------------------------------------
def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["group"] = df["station"].map(STATION_GROUP)

    # 일 단위로 집계 (원시 시간단위 420,000행은 단일 코어 MCMC에는 과함;
    # 계층모델의 목적은 station/그룹 구조 파악이므로 daily mean으로 충분)
    daily = (
        df.groupby(["station", "group", pd.Grouper(key="datetime", freq="D")])
        .agg({**{f: "mean" for f in FEATURES}, "PM2.5": "mean", "RAIN": "sum"})
        .reset_index()
    )
    daily = daily.dropna(subset=["PM2.5"] + FEATURES)

    # log 변환 (오른쪽 꼬리 긴 분포 -> Normal likelihood 가정에 적합하게)
    daily["log_pm25"] = np.log1p(daily["PM2.5"])

    # 예측변수 표준화 (station 간 계수 비교 및 수렴 안정성을 위해 필수)
    for f in FEATURES:
        daily[f + "_z"] = (daily[f] - daily[f].mean()) / daily[f].std()

    return daily


def build_indices(daily):
    stations = sorted(daily["station"].unique())
    groups = sorted(daily["group"].unique())
    station_idx = daily["station"].map({s: i for i, s in enumerate(stations)}).values
    group_of_station = np.array(
        [groups.index(STATION_GROUP[s]) for s in stations]
    )
    return stations, groups, station_idx, group_of_station


# ----------------------------------------------------------------------
# 2. 모델 정의: 계층(부분 풀링) / 완전 풀링 / 완전 비풀링 비교용
# ----------------------------------------------------------------------
def build_hierarchical_model(daily, stations, groups, station_idx, group_of_station):
    n_station = len(stations)
    n_group = len(groups)
    X = daily[[f + "_z" for f in FEATURES]].values
    y = daily["log_pm25"].values

    coords = {"station": stations, "group": groups, "feature": FEATURES, "obs_id": daily.index}

    with pm.Model(coords=coords) as model:
        station_idx_ = pm.Data("station_idx", station_idx, dims="obs_id")
        group_of_station_ = pm.Data("group_of_station", group_of_station, dims="station")
        X_ = pm.Data("X", X, dims=("obs_id", "feature"))
        y_ = pm.Data("y_obs", y, dims="obs_id")

        # -- 최상위(global) --
        mu_global = pm.Normal("mu_global", mu=4.0, sigma=1.0)  # log(PM2.5) 대략 중심값
        tau_group = pm.HalfNormal("tau_group", sigma=1.0)

        # -- 그룹 레벨 절편 (non-centered) --
        group_offset = pm.Normal("group_offset", 0, 1, dims="group")
        mu_group = pm.Deterministic("mu_group", mu_global + tau_group * group_offset, dims="group")

        # -- station 레벨 절편 (non-centered, 그룹에 부분 풀링) --
        tau_station = pm.HalfNormal("tau_station", sigma=1.0)
        station_offset = pm.Normal("station_offset", 0, 1, dims="station")
        alpha_station = pm.Deterministic(
            "alpha_station",
            mu_group[group_of_station_] + tau_station * station_offset,
            dims="station",
        )

        # -- 기상변수 계수: station별로 부분 풀링 (계수도 station마다 조금씩 다르되 공유) --
        beta_global = pm.Normal("beta_global", 0, 1, dims="feature")
        tau_beta = pm.HalfNormal("tau_beta", sigma=0.5, dims="feature")
        beta_offset = pm.Normal("beta_offset", 0, 1, dims=("station", "feature"))
        beta_station = pm.Deterministic(
            "beta_station", beta_global + tau_beta * beta_offset, dims=("station", "feature")
        )

        mu = alpha_station[station_idx_] + (X_ * beta_station[station_idx_]).sum(axis=-1)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        pm.Normal("obs", mu=mu, sigma=sigma_obs, observed=y_, dims="obs_id")

    return model


def build_pooled_model(daily):
    """완전 풀링: station 구분 없이 하나의 회귀식."""
    X = daily[[f + "_z" for f in FEATURES]].values
    y = daily["log_pm25"].values
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 4.0, 1.0)
        beta = pm.Normal("beta", 0, 1, shape=len(FEATURES))
        sigma = pm.HalfNormal("sigma", 1.0)
        mu = alpha + (X * beta).sum(axis=-1)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model


def build_unpooled_model(daily, stations, station_idx):
    """완전 비풀링: station마다 완전히 독립적인 파라미터."""
    n_station = len(stations)
    X = daily[[f + "_z" for f in FEATURES]].values
    y = daily["log_pm25"].values
    coords = {"station": stations, "feature": FEATURES}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 4.0, 1.0, dims="station")
        beta = pm.Normal("beta", 0, 1, dims=("station", "feature"))
        sigma = pm.HalfNormal("sigma", 1.0)
        mu = alpha[station_idx] + (X * beta[station_idx]).sum(axis=-1)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model


# ----------------------------------------------------------------------
# 3. 실행
# ----------------------------------------------------------------------
def main():
    print("[1/6] 데이터 로드 및 일단위 집계...")
    daily = load_and_prepare(INPUT_PATH)
    stations, groups, station_idx, group_of_station = build_indices(daily)
    print(f"  -> {len(daily)} station-day rows, {len(stations)} stations, {len(groups)} groups")

    sample_kwargs = dict(
        draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
        target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True,
    )

    print("[2/6] 계층모델(hierarchical) 샘플링...")
    hier_model = build_hierarchical_model(daily, stations, groups, station_idx, group_of_station)
    with hier_model:
        idata_hier = pm.sample(**sample_kwargs)
        pm.compute_log_likelihood(idata_hier)  # LOO 비교에 필요
        pm.sample_posterior_predictive(idata_hier, extend_inferencedata=True, random_seed=RANDOM_SEED)

    print("[3/6] 완전 풀링(pooled) 모델 샘플링...")
    pooled_model = build_pooled_model(daily)
    with pooled_model:
        idata_pooled = pm.sample(**sample_kwargs)
        pm.compute_log_likelihood(idata_pooled)

    print("[4/6] 완전 비풀링(unpooled) 모델 샘플링...")
    unpooled_model = build_unpooled_model(daily, stations, station_idx)
    with unpooled_model:
        idata_unpooled = pm.sample(**sample_kwargs)
        pm.compute_log_likelihood(idata_unpooled)

    # -- 수렴 진단 --
    print("[5/6] 수렴 진단 및 모델 비교...")
    summary = az.summary(idata_hier, var_names=["mu_global", "tau_group", "tau_station",
                                                 "beta_global", "tau_beta", "sigma_obs"])
    print(summary)
    max_rhat = summary["r_hat"].max()
    if max_rhat > 1.01:
        print(f"  [경고] r_hat 최대값 {max_rhat:.3f} > 1.01 -> tune/draws를 늘려 재실행 권장")
    else:
        print(f"  수렴 양호: r_hat 최대값 {max_rhat:.3f}")

    az.plot_trace(idata_hier, var_names=["mu_global", "tau_group", "tau_station", "beta_global"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trace_diagnostics.png"), dpi=150)
    plt.close()

    # -- LOO 비교: 정보 공유가 실제로 이득인지 정량적으로 확인 --
    try:
        cmp_df = az.compare({
            "hierarchical": idata_hier,
            "pooled": idata_pooled,
            "unpooled": idata_unpooled,
        })
        cmp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
        print(cmp_df)
    except Exception as e:
        print(f"  [주의] LOO 비교 계산 중 오류 (log_likelihood 그룹 확인 필요): {e}")

    # -- station별 절편 forest plot --
    az.plot_forest(idata_hier, var_names=["alpha_station"], combined=True)
    plt.title("Station별 log(PM2.5) 절편 사후분포 (94% HDI)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "station_intercepts_forest.png"), dpi=150)
    plt.close()

    # -- posterior predictive check --
    # (arviz>=1.0 에서 plot_ppc -> plot_ppc_dist 로 개편됨)
    az.plot_ppc_dist(idata_hier, num_samples=100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ppc_check.png"), dpi=150)
    plt.close()

    # -- station/그룹 파라미터 요약 저장 --
    print("[6/6] 결과 저장...")
    full_summary = az.summary(idata_hier, var_names=["alpha_station", "mu_group", "beta_station"])
    full_summary.to_csv(os.path.join(OUTPUT_DIR, "posterior_summary.csv"))

    print(f"완료. 결과는 {OUTPUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
