"""For each gate drop list, drop the most common concepts until the same share of
the index is removed.

python -m p4.df_drops --df idx/full/df.npy --gates "gate_lam*.drop.npy"
"""
import argparse
import glob

import numpy as np


def share_removed(df, drop):
    """Fraction of all postings that belong to the dropped concepts."""
    return df[drop].sum() / df.sum()


def most_common(df, share):
    """Most common concepts, enough of them to remove `share` of the postings."""
    order = np.argsort(-df, kind="stable")
    cumulative = np.cumsum(df[order]) / df.sum()
    n = np.searchsorted(cumulative, share) + 1
    return order[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", required=True)
    parser.add_argument("--gates", required=True)
    args = parser.parse_args()

    df = np.load(args.df).astype(np.float64)
    for path in sorted(glob.glob(args.gates)):
        gate = np.load(path)
        share = share_removed(df, gate)
        drop = most_common(df, share)
        out = path.replace("gate_", "df_")
        np.save(out, drop)
        print(path, len(gate), "gate concepts", f"{share:.1%} removed", len(drop), "DF concepts")


if __name__ == "__main__":
    main()