import numpy as np
import orjson
import os

NUM_CONCEPTS = 24576
MAX_LATENTS = 24
TEST_DIR = "p4/test"                         
DF_PATH = os.path.join(TEST_DIR, "df_b0.npy")  

def load_or_compute_df(latents_path):     
    if os.path.exists(DF_PATH):
        print("loading saved df from", DF_PATH)
        return np.load(DF_PATH)
    df = concept_df(latents_path)
    os.makedirs(TEST_DIR, exist_ok=True)
    np.save(DF_PATH, df)
    print("saved df to", DF_PATH)
    return df

def concept_df(path, max_latents=MAX_LATENTS, num_concepts=NUM_CONCEPTS):
    df = np.zeros(num_concepts, dtype=np.int64)
    with open(path) as f:
        for line in f:
            record = orjson.loads(line)
            ids, weights = np.asarray(record["ids"]), np.asarray(record["weight"], dtype=np.float64)
            keep = np.argsort(-weights, kind = 'stable')[:max_latents]
            keep_ids = ids[keep]
            df[keep_ids] += 1
    return df

def top_common(df, n):
    """Return the ids of the n most common concepts (concepts that never appear are excluded)."""
    used = np.flatnonzero(df)                 
    relevant_df = df[used]
    top_used_positions = np.argsort(-relevant_df, kind = 'stable')[:n]
    return used[top_used_positions]

def postings_removed(df, drop_ids):
    """Fraction of all index entries that belong to the dropped concepts."""
    df_sum = np.sum(df)
    df_dropped_ids_sum = np.sum(df[drop_ids])
    return df_dropped_ids_sum/df_sum

def n_for_fraction(df, target):
    """Smallest number of most-common concepts whose postings reach `target` share of the index."""
    used = np.flatnonzero(df)
    sorted_df = np.sort(df[used])[::-1]               
    cumulative = np.cumsum(sorted_df) / df.sum()       
    return int(np.searchsorted(cumulative, target) + 1)


if __name__ == "__main__":
    PATH = "/Users/eelmeal/Desktop/sae-retrieval/data/passages.latent.jsonl"
    df = load_or_compute_df(PATH)            
    top = top_common(df, 10)
    print(postings_removed(df, top_common(df, 1))) 

    for target in [0.044, 0.10, 0.25, 0.50]:
        n = n_for_fraction(df, target)
        drop = top_common(df, n)
        path = f"{TEST_DIR}/drop_df_{int(target * 100)}pct.npy"
        np.save(path, drop)
        print(f"{target:.0%}: drop {n} concepts → {postings_removed(df, drop):.1%} removed → {path}")