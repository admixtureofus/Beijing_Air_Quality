"""
베이징 대기질(PRSA) 데이터 전처리 스크립트
12개 관측소 CSV를 통합하고, datetime 생성, 결측치 처리, 파생변수 생성까지 수행한다.
결과물: cleaned_prsa_all.csv (팀 전체가 공용으로 쓸 수 있는 정제 데이터)
"""

import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ------------------------------------------------------------------
# 1. 12개 관측소 CSV 자동 로드 & 통합
# ------------------------------------------------------------------
DATA_PATH = r"C:\Users\과표사업단\Documents\beijing\prsa_all.csv"   # 예: /mnt/project/prsa_all.csv

df = pd.read_csv(DATA_PATH)
print(f"통합 전체 shape: {df.shape}")
print(f"관측소 수: {df['station'].nunique()}, 관측소 목록: {sorted(df['station'].unique())}")

# ------------------------------------------------------------------
# 2. datetime 컬럼 생성 (year, month, day, hour 조합)
# ------------------------------------------------------------------
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

# ------------------------------------------------------------------
# 3. 결측치 현황 확인
# ------------------------------------------------------------------
print("\n[결측치 개수 / 비율]")
na_summary = pd.DataFrame({
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(2),
})
print(na_summary[na_summary["n_missing"] > 0])


# # 패턴 시각화 (결측치 위치 확인용)
# FIG_DIR = r"C:\Users\과표사업단\Documents\beijing\outputs\figures"

# pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
# weather = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# sns.heatmap(df[pollutants + weather].isna(), cbar=False, cmap="viridis")
# plt.title("pattern of missing values (yellow=missing, purple=present)")
# plt.tight_layout()
# plt.savefig(f"{FIG_DIR}/01_missing_values.png", dpi=120)
# plt.close()


# ------------------------------------------------------------------
# 4. 결측치 처리
#    - 오염물질/기상 수치형 : 관측소별 시계열 순서로 선형보간 (limit=6시간)
#      그래도 남는 결측(구간 시작/끝 등)은 관측소별 중앙값으로 대체
#    - wd(풍향, 범주형) : 관측소별 최빈값으로 대체
# ------------------------------------------------------------------
numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
"TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# 문자열로 읽힌 컬럼(원본에 결측이 섞여 object/str 타입일 수 있음) 숫자형 변환
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

interpolated_parts = []
for station_name, g in df.groupby("station"):
    g = g.set_index("datetime")
    g[numeric_cols] = g[numeric_cols].interpolate(method="time", limit=6, limit_direction="both")
    g = g.reset_index()
    g["station"] = station_name  # groupby apply 과정에서 유실되지 않도록 명시적으로 보존
    interpolated_parts.append(g)
df = pd.concat(interpolated_parts, ignore_index=True)

# 보간 후에도 남은 결측 -> 관측소별 중앙값 대체
for col in numeric_cols:
    df[col] = df.groupby("station")[col].transform(lambda s: s.fillna(s.median()))

# 풍향(wd) 결측 -> 관측소별 최빈값
df["wd"] = df.groupby("station")["wd"].transform(
    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "N")
)

print("\n[처리 후 잔여 결측치 총합]:", df[numeric_cols + ["wd"]].isna().sum().sum())

# ------------------------------------------------------------------
# 5. 파생변수 생성
# ------------------------------------------------------------------
# 풍향(16방위 문자열) -> 각도 -> sin/cos 순환 인코딩
wd_to_deg = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
    "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}
df["wd_deg"] = df["wd"].map(wd_to_deg)
df["wd_sin"] = np.sin(np.radians(df["wd_deg"]))
df["wd_cos"] = np.cos(np.radians(df["wd_deg"]))

# 계절 / 요일 / 주말여부 / 시간대 파생
df["season"] = df["month"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                 3: "Spring", 4: "Spring", 5: "Spring",
                                 6: "Summer", 7: "Summer", 8: "Summer",
                                 9: "Fall", 10: "Fall", 11: "Fall"})
df["dow"] = df["datetime"].dt.day_name()
df["is_weekend"] = df["datetime"].dt.weekday >= 5

# 중국 기준 PM2.5 등급 (참고용, 필요시 WHO 기준으로 교체 가능)
def pm25_grade(x):
    if pd.isna(x):
        return np.nan
    if x <= 35:
        return "Good"
    elif x <= 75:
        return "Moderate"
    elif x <= 115:
        return "Lightly Polluted"
    elif x <= 150:
        return "Moderately Polluted"
    elif x <= 250:
        return "Heavily Polluted"
    else:
        return "Severely Polluted"

df["pm25_grade"] = df["PM2.5"].apply(pm25_grade)


