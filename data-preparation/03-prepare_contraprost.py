import pandas as pd
import sys

lang = sys.argv[1]


df = pd.read_csv(f"./data/en_{lang}.csv", index_col=0)

rows = []
for _, row in df.iterrows():
    src = row["sentence"]
    t1, t2 = row["translation_1"], row["translation_2"]
    a1, a2 = row["audio_1"], row["audio_2"]
    rows.extend([
        {"src": src, "mt": t1, "score": 100, "src_audio": a1},
        {"src": src, "mt": t2, "score": 0,   "src_audio": a1},
        {"src": src, "mt": t1, "score": 0,   "src_audio": a2},
        {"src": src, "mt": t2, "score": 100, "src_audio": a2},
    ])

out = pd.DataFrame(rows, columns=["src", "mt", "score", "src_audio"])
out.to_csv(f"./data/en_{lang}_expanded.csv", index=False)
print(f"Done: {len(df)} input rows -> {len(out)} output rows")
