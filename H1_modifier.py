import pandas as pd

INPUT_CSV = "FILE_DUMP_EURUSD_M1.csv"
OUTPUT_CSV = "EURUSD_H1.csv"

print("📥 Loading M1 data...")
df = pd.read_csv(INPUT_CSV, header=None)

df.columns = ["date", "time", "open", "high", "low", "close", "volume"]

df["datetime"] = pd.to_datetime(
    df["date"] + " " + df["time"],
    format="%Y.%m.%d %H:%M"
)
df.set_index("datetime", inplace=True)

for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

print("🔄 Resampling to H1...")
h1 = df.resample("1h").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
})
h1 = h1.dropna()

h1["date"] = h1.index.strftime("%Y.%m.%d")
h1["time"] = h1.index.strftime("%H:%M")
h1 = h1[["date", "time", "open", "high", "low", "close", "volume"]]

h1.to_csv(OUTPUT_CSV, index=False, header=False)
print(f"✅ Conversion complete → {OUTPUT_CSV}")
