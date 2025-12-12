from datasets import load_dataset
import pandas as pd
from pathlib import Path

def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "rough4q_full.csv"

    ds = load_dataset("monetjoe/EMelodyGen", "Rough4Q")

    dfs = []
    for split_name, split_ds in ds.items():
        df_split = split_ds.to_pandas()
        df_split["split"] = split_name
        dfs.append(df_split)

    df_all = pd.concat(dfs, ignore_index=True)

    label_map = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"}
    if "label" in df_all.columns:
        df_all["label_name"] = df_all["label"].map(label_map)

    df_all = df_all.rename(columns={"data": "abc", "label": "solution"})

    keep = [c for c in ["split", "solution", "label_name", "valence", "arousal", "prompt", "abc"] if c in df_all.columns]
    df_all = df_all[keep]

    if "abc" not in df_all.columns or "solution" not in df_all.columns:
        raise RuntimeError(f"Expected columns not found. Available columns: {list(df_all.columns)}")

    df_all = df_all.dropna(subset=["abc", "solution"])
    df_all["abc"] = df_all["abc"].astype(str)
    df_all["solution"] = df_all["solution"].astype(int)

    df_all.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} (rows={len(df_all)})")
    print("Columns:", list(df_all.columns))

if __name__ == "__main__":
    main()
