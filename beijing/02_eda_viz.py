"""
베이징 대기질(PRSA) EDA + 시각화 스크립트
01_preprocess.py 로 만든 cleaned_prsa_all.csv 를 입력으로 사용한다.
그림은 /mnt/user-data/outputs/figures 에 저장한다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False 
#____________________________________________________________________________________________________

sns.set_theme(style="whitegrid", font_scale=0.9)
FIG_DIR = r"파일 경로"

df = pd.read_csv(r"파일 경로", parse_dates=["datetime"])
print(f"데이터 shape: {df.shape}, 관측소 수: {df['station'].nunique()}")

pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
weather = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# ------------------------------------------------------------------
# 1. 기초 통계 요약 (관측소별)
# ------------------------------------------------------------------
summary = df.groupby("station")[pollutants].agg(["mean", "median", "std"]).round(1)
summary.to_csv(f"{FIG_DIR}/../station_pollutant_summary.csv")
print("\n[관측소별 오염물질 요약] -> station_pollutant_summary.csv 저장")

# ------------------------------------------------------------------
# 2. 결측 패턴 히트맵 (원본 대비 처리 확인용 - 여기선 전체 분포 확인)
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.heatmap(df[pollutants + weather].isna(), cbar=False, cmap="viridis")
plt.title("pattern of missing values (yellow=missing, purple=present)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_missing_pattern_after_processing.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 3. 오염물질 분포 (히스토그램, station 무관 전체)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, pollutants):
    sns.histplot(df[col], bins=60, ax=ax, kde=True, color="steelblue")
    ax.set_title(f"{col} pollutant distribution")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_pollutant_distributions.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 4. 관측소별 PM2.5 비교 (Boxplot)
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))
order = df.groupby("station")["PM2.5"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="station", y="PM2.5", order=order, showfliers=False, palette="coolwarm")
plt.xticks(rotation=45, ha="right")
plt.title("pm2.5 distribution by station (outliers removed)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_station_pm25_boxplot.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 5. 월별 PM2.5 추세 (연도별 라인)
# ------------------------------------------------------------------
monthly = df.groupby([df["datetime"].dt.to_period("M")])["PM2.5"].mean()
plt.figure(figsize=(13, 5))
monthly.index = monthly.index.to_timestamp()
plt.plot(monthly.index, monthly.values, marker="o", markersize=3, color="firebrick")
plt.title("monthly average PM2.5 trend (2013-2017)")
plt.ylabel("PM2.5 (µg/m³)")
plt.xlabel("year-month")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_monthly_pm25_trend.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 6. 계절별 x 시간대별 PM2.5 히트맵 (일중 패턴)
# ------------------------------------------------------------------
pivot = df.pivot_table(values="PM2.5", index="hour", columns="season", aggfunc="mean")
pivot = pivot[["Spring", "Summer", "Fall", "Winter"]]
plt.figure(figsize=(7, 6))
sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f")
plt.title("season x hour-wise average PM2.5")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_season_hour_heatmap.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 7. 오염물질-기상변수 상관관계 히트맵
# ------------------------------------------------------------------
corr = df[pollutants + weather + ["wd_sin", "wd_cos"]].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("pollutants & weather variables correlation")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_correlation_heatmap.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 8. 주중 vs 주말 PM2.5 비교
# ------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="is_weekend", y="PM2.5", showfliers=False, palette="Set2")
plt.xticks([0, 1], ["weekday", "weekend"])
plt.title("weekday vs weekend PM2.5 comparison")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_weekday_weekend_pm25.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 9. PM2.5 등급 분포 (관측소별 stacked bar)
# ------------------------------------------------------------------
grade_order = ["Good", "Moderate", "Lightly Polluted", "Moderately Polluted",
               "Heavily Polluted", "Severely Polluted"]
grade_pct = (
    df.groupby("station")["pm25_grade"]
    .value_counts(normalize=True)
    .unstack()
    .reindex(columns=grade_order)
    * 100
)
grade_pct.plot(kind="bar", stacked=True, figsize=(12, 6),
                colormap="RdYlGn_r")
plt.ylabel("Percentage (%)")
plt.title("PM2.5 Grade Distribution by Station")
plt.xticks(rotation=45, ha="right")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_pm25_grade_by_station.png", dpi=120)
plt.close()

print(f"\n총 {len(list(__import__('pathlib').Path(FIG_DIR).glob('*.png')))}개 그림 저장 완료 -> {FIG_DIR}")
