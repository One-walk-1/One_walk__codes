import os
import random
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional

def _fmt_index_str(ix, width: Optional[int]=None) -> str:
    s = str(ix)
    if s.isdigit():
        n = int(s)
        if width is None:
            width = max(5, len(str(n)))
        return str(n).zfill(width)
    else:
        if width is None:
            width = max(5, len(s))
        return s.zfill(width)

def _unique_tuples(series_tuple: Iterable[Tuple]) -> list:
    seen = set()
    out = []
    for t in series_tuple:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def split_dataset_by_rx_man(
    datadir: str,
    csv_name: str = "spectrum_info.csv",
    spectrums_subdir: str = "spectrums",
    rx_cols: Tuple[str, str, str] = ("rx_x","rx_y","rx_z"),
    man_cols: Tuple[str, str, str] = ("man_x","man_y","man_z"),
    rx_ratio: float = 0.8,
    man_ratio: float = 0.8,
    seed: int = 42,
    position_precision: Optional[int] = None,  
    verbose: bool = False
):
    rng = random.Random(seed)
    np.random.seed(seed)

    csv_path = os.path.join(datadir, csv_name)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Not found {csv_path}")

    df = pd.read_csv(csv_path)
    if position_precision is not None:
        for c in list(rx_cols) + list(man_cols):
            df[c] = df[c].round(position_precision)

    spectrum_dir = os.path.join(datadir, spectrums_subdir)
    if os.path.isdir(spectrum_dir):
        png_set = {os.path.splitext(f)[0] for f in os.listdir(spectrum_dir) if f.lower().endswith(".png")}
        before = len(df)
        df = df[df["index"].astype(str).str.zfill(5).isin(png_set)].copy()
        if verbose:
            print(f"Filtered samples without corresponding images: {before} -> {len(df)}")

    df["rx_key"]  = list(zip(df[rx_cols[0]], df[rx_cols[1]], df[rx_cols[2]]))
    df["man_key"] = list(zip(df[man_cols[0]], df[man_cols[1]], df[man_cols[2]]))

    rx_unique  = list(set(df["rx_key"].tolist()))
    man_unique = list(set(df["man_key"].tolist()))

    rx_train_n  = max(1, int(round(len(rx_unique)  * rx_ratio)))
    man_train_n = max(1, int(round(len(man_unique) * man_ratio)))

    rx_train_set  = set(rng.sample(rx_unique,  rx_train_n)) if rx_train_n < len(rx_unique) else set(rx_unique)
    man_train_set = set(rng.sample(man_unique, man_train_n)) if man_train_n < len(man_unique) else set(man_unique)

    rx_test_set   = set(rx_unique)  - rx_train_set
    man_test_set  = set(man_unique) - man_train_set

    def assign_split(row):
        rx_key, man_key = row["rx_key"], row["man_key"]
        if (rx_key in rx_train_set) and (man_key in man_train_set):
            return "train"
        elif (rx_key in rx_test_set) and (man_key in man_test_set):
            return "test"
        else:
            return "val"

    df["split"] = df.apply(assign_split, axis=1)

    try:
        width = max(5, len(str(int(df["index"].max()))))
    except Exception:
        width = 5
    df["index_str"] = df["index"].astype(str).str.zfill(width)

    out_train = df.loc[df["split"]=="train", "index_str"].tolist()
    out_test  = df.loc[df["split"]=="test",  "index_str"].tolist()
    out_val   = df.loc[df["split"]=="val",   "index_str"].tolist()

    np.savetxt(os.path.join(datadir, "train_index.txt"), np.array(out_train, dtype=str), fmt="%s")
    np.savetxt(os.path.join(datadir, "test_index.txt"),  np.array(out_test,  dtype=str), fmt="%s")
    np.savetxt(os.path.join(datadir, "val_index.txt"),   np.array(out_val,   dtype=str), fmt="%s")
