"""
01_preprocess.py  (v2)
======================
베이징 대기질(PRSA) 데이터 전처리 스크립트
12개 관측소 CSV를 통합하고, datetime 생성, 결측치 처리, 파생변수 생성까지 수행한다.

v1 대비 변경점
--------------
- v1은 docstring에 "결과물: cleaned_prsa_all.csv" 라고 적혀 있었지만 실제로 파일을
  저장하는 코드가 없었다. OUTPUT_PATH 및 to_csv 저장 단계를 추가했다.
  (03/04/05 스크립트가 모두 이 파일을 입력으로 쓰므로 재실행 전 반드시 필요)
- 03/04/05에서 잔차 AR(1)을 쓰려면 station별 시간축이 정렬·중복 없이 유지되어야 하므로,
  저장 직전에 (station, datetime) 중복 제거 및 정렬을 명시적으로 수행한다.
- 결측 처리 전후 요약을 CSV로 남겨 보고서에 그대로 인용할 수 있게 했다.
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 0. 경로 설정
# ------------------------------------------------------------------
DATA_PATH = r"파일 경로"                      # 12개 관측소가 합쳐진 원본 CSV
OUTPUT_PATH = r"저장경로\cleaned_prsa_all.csv"
NA_REPORT_PATH = r"저장경로\missing_summary.csv"

# ------------------------------------------------------------------
# 1. 로드 & 기본 확인
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"통합 전체 shape: {df.shape}")
print(f"관측소 수: {df['station'].nunique()}, 목록: {sorted(df['station'].unique())}")

# ------------------------------------------------------------------
# 2. datetime 컬럼 생성
# ------------------------------------------------------------------
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

# ------------------------------------------------------------------
# 3. 결측치 현황
# ------------------------------------------------------------------
na_summary = pd.DataFrame({
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(2),
})
print("\n[결측치 개수 / 비율]")
print(na_summary[na_summary["n_missing"] > 0])
na_summary.to_csv(NA_REPORT_PATH, encoding="utf-8-sig")

# ------------------------------------------------------------------
# 4. 결측치 처리
#    - 수치형: 관측소별 시간 선형보간(limit=6시간) -> 잔여분은 관측소별 중앙값
#    - wd(풍향, 범주형): 관측소별 최빈값
# ------------------------------------------------------------------
numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

interpolated_parts = []
for station_name, g in df.groupby("station"):
    g = g.set_index("datetime")
    g[numeric_cols] = g[numeric_cols].interpolate(method="time", limit=6, limit_direction="both")
    g = g.reset_index()
    g["station"] = station_name
    interpolated_parts.append(g)
df = pd.concat(interpolated_parts, ignore_index=True)

for col in numeric_cols:
    df[col] = df.groupby("station")[col].transform(lambda s: s.fillna(s.median()))

df["wd"] = df.groupby("station")["wd"].transform(
    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "N")
)

print("\n[처리 후 잔여 결측치 총합]:", df[numeric_cols + ["wd"]].isna().sum().sum())

# ------------------------------------------------------------------
# 5. 파생변수
# ------------------------------------------------------------------
wd_to_deg = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
    "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}
df["wd_deg"] = df["wd"].map(wd_to_deg)
df["wd_sin"] = np.sin(np.radians(df["wd_deg"]))
df["wd_cos"] = np.cos(np.radians(df["wd_deg"]))

df["season"] = df["month"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                3: "Spring", 4: "Spring", 5: "Spring",
                                6: "Summer", 7: "Summer", 8: "Summer",
                                9: "Fall", 10: "Fall", 11: "Fall"})
df["dow"] = df["datetime"].dt.day_name()
df["is_weekend"] = df["datetime"].dt.weekday >= 5


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

# ------------------------------------------------------------------
# 6. 시계열 정합성 확인 후 저장
#    (AR(1) 모형은 station별 시간축의 정렬과 중복 없음을 전제로 한다)
# ------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=["station", "datetime"]).sort_values(
    ["station", "datetime"]).reset_index(drop=True)
if before != len(df):
    print(f"[정리] (station, datetime) 중복 {before - len(df)}행 제거")

gap_report = (
    df.groupby("station")["datetime"]
    .apply(lambda s: (s.diff().dt.total_seconds().div(3600).dropna() != 1).sum())
)
print("\n[station별 1시간 간격이 아닌 구간 수]")
print(gap_report)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n저장 완료 -> {OUTPUT_PATH}  (shape: {df.shape})")
