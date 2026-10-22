import numpy as np
from itertools import combinations

def discover_associations(X, feature_names=None, max_degree=3, min_support=3,
                          min_probability=0.5, min_lift=1.0, top_k=25):
    """Discover antecedent -> target rules from a binary (rows x features) matrix."""
    X = np.asarray(X, dtype=np.uint8)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a non-empty 2D array.")
    if not np.isin(X, [0, 1]).all():
        raise ValueError("X must contain only 0 and 1.")
    n_rows, n_features = X.shape
    if n_features < 2:
        raise ValueError("At least two features are required.")
    if max_degree < 1 or min_support < 1:
        raise ValueError("max_degree and min_support must be positive.")
    names = list(feature_names or [f"F{i}" for i in range(n_features)])
    if len(names) != n_features:
        raise ValueError("feature_names length must equal the number of columns.")

    # Each feature becomes a bitset over observations: bit r is 1 if X[r, j] == 1.
    # Python integers support bitsets larger than 64 bits.
    feature_masks = []
    for j in range(n_features):
        mask = 0
        for r in np.flatnonzero(X[:, j]):
            mask |= 1 << int(r)
        feature_masks.append(mask)

    baseline = X.mean(axis=0)
    results = []
    max_degree = min(max_degree, n_features - 1)

    for degree in range(1, max_degree + 1):
        degree_results = []
        for antecedent in combinations(range(n_features), degree):
            active_mask = (1 << n_rows) - 1
            for j in antecedent:
                active_mask &= feature_masks[j]
                if active_mask == 0:
                    break

            support = active_mask.bit_count()
            if support < min_support:
                continue

            for target in range(n_features):
                if target in antecedent:
                    continue
                hits = (active_mask & feature_masks[target]).bit_count()
                probability = hits / support
                base = float(baseline[target])
                lift = probability / base if base > 0 else float("inf")
                if probability >= min_probability and lift >= min_lift:
                    degree_results.append({
                        "degree": degree,
                        "antecedent": tuple(names[i] for i in antecedent),
                        "target": names[target],
                        "support": support,
                        "hits": hits,
                        "probability": probability,
                        "baseline": base,
                        "lift": lift,
                    })
        degree_results.sort(
            key=lambda r: (r["lift"], r["probability"], r["support"]),
            reverse=True
        )
        results.extend(degree_results[:top_k])

    return results

def make_demo_data(seed=7):
    """Synthetic example with a planted degree-3 association."""
    rng = np.random.default_rng(seed)
    n = 500
    rates = np.array([.35, .30, .25, .20, .18, .15, .12, .10])
    X = (rng.random((n, len(rates))) < rates).astype(np.uint8)
    trigger = (X[:, 0] & X[:, 1] & X[:, 2]).astype(bool)
    X[trigger, 7] = (rng.random(trigger.sum()) < .90).astype(np.uint8)
    return X

if __name__ == "__main__":
    names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    X = make_demo_data()
    rules = discover_associations(
        X, feature_names=names, max_degree=3, min_support=3,
        min_probability=0.50, min_lift=1.5, top_k=15
    )

    print(f"Data: {X.shape[0]} observations x {X.shape[1]} features")
    print("Example row bitmask:", format(sum(int(X[0, j]) << j for j in range(X.shape[1])),
                                         f"0{X.shape[1]}b"))
    print("\nTop discovered associations:")
    if not rules:
        print("No rules passed the thresholds; lower min_support/min_probability/min_lift.")
    for r in rules:
        lhs = " & ".join(r["antecedent"])
        print(f"degree={r['degree']} | {lhs} -> {r['target']} | "
              f"P={r['probability']:.3f} | baseline={r['baseline']:.3f} | "
              f"lift={r['lift']:.2f}x | support={r['support']}")
