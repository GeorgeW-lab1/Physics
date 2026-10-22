import numpy as np
from itertools import combinations

def build_non_association_matrix(
    X,
    feature_names=None,
    rules=None,
    min_support=5,
    min_abs_lift_deviation=0.10,
    smoothing=1.0,
    seed=42,
    permutations=200,
):
    """
    Build a programmable association / chance / non-association matrix.

    X: binary array (observations, features).
    rules: optional list of dicts, e.g.
      {"if": ["A", "B"], "then": "C", "chance": 0.8, "weight": 1.0}
    min_support: minimum co-occurrence count for a pair.
    min_abs_lift_deviation: minimum |P(B|A)-P(B)| to label a pair associated.
    smoothing: Laplace smoothing for conditional probabilities.
    permutations: random row-shuffle tests for pairwise chance calibration.
    """
    X = np.asarray(X, dtype=np.uint8)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a non-empty 2D array.")
    if not np.isin(X, [0, 1]).all():
        raise ValueError("X must contain only 0 and 1.")
    n, m = X.shape
    if m < 2:
        raise ValueError("At least two features are required.")

    names = list(feature_names or [f"F{i}" for i in range(m)])
    if len(names) != m or len(set(names)) != m:
        raise ValueError("Feature names must be unique and match X columns.")
    index = {name: i for i, name in enumerate(names)}
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    # Pairwise outputs:
    score = np.zeros((m, m), dtype=float)
    probability = np.full((m, m), np.nan, dtype=float)
    lift = np.full((m, m), np.nan, dtype=float)
    support = np.zeros((m, m), dtype=int)
    status = np.full((m, m), "diagonal", dtype=object)
    chance_p = np.full((m, m), np.nan, dtype=float)
    rng = np.random.default_rng(seed)

    rates = X.mean(axis=0)
    for i in range(m):
        for j in range(i + 1, m):
            both = int(np.sum(X[:, i] & X[:, j]))
            count_i = int(X[:, i].sum())
            count_j = int(X[:, j].sum())
            support[i, j] = support[j, i] = both

            # P(j|i), smoothed toward baseline P(j).
            p_j_given_i = (both + smoothing * rates[j]) / (count_i + smoothing) if count_i + smoothing else rates[j]
            p_i_given_j = (both + smoothing * rates[i]) / (count_j + smoothing) if count_j + smoothing else rates[i]
            probability[i, j] = p_j_given_i
            probability[j, i] = p_i_given_j

            # Symmetric association effect: observed joint rate vs expected joint rate.
            expected = rates[i] * rates[j]
            observed = both / n
            denom = min(expected, rates[i] * (1 - rates[j]), rates[j] * (1 - rates[i]),
                        (1 - rates[i]) * (1 - rates[j]))
            if denom > 0:
                raw = (observed - expected) / denom
            else:
                raw = 0.0
            raw = float(np.clip(raw, -1.0, 1.0))
            score[i, j] = score[j, i] = raw

            ratio = observed / expected if expected > 0 else np.nan
            lift[i, j] = lift[j, i] = ratio

            # Permutation null: independently shuffle one feature to preserve each marginal rate.
            if permutations > 0 and count_i > 0 and count_i < n and count_j > 0 and count_j < n:
                observed_abs = abs(observed - expected)
                exceed = 0
                for _ in range(permutations):
                    shuffled = rng.permutation(X[:, j])
                    null_diff = abs(float(np.mean(X[:, i] * shuffled)) - expected)
                    exceed += null_diff >= observed_abs
                pval = (exceed + 1) / (permutations + 1)
            else:
                pval = np.nan
            chance_p[i, j] = chance_p[j, i] = pval

            effect = abs(p_j_given_i - rates[j])
            if both < min_support:
                label = "insufficient_data"
            elif effect < min_abs_lift_deviation:
                label = "non_association"
            elif np.isfinite(pval) and pval > 0.05:
                label = "chance_or_uncertain"
            else:
                label = "positive_association" if raw > 0 else "negative_association"
            status[i, j] = status[j, i] = label

    # Programmable higher-order rules adjust a separate directed matrix.
    rule_matrix = np.zeros((m, m), dtype=float)
    rule_log = []
    for rule in (rules or []):
        antecedent = rule.get("if", [])
        target = rule.get("then")
        chance = float(rule.get("chance", 0.5))
        weight = float(rule.get("weight", 1.0))
        if not antecedent or target not in index or any(a not in index for a in antecedent):
            raise ValueError(f"Invalid rule: {rule}")
        if not 0 <= chance <= 1 or weight < 0:
            raise ValueError(f"Rule chance must be in [0,1] and weight >= 0: {rule}")
        cols = [index[a] for a in antecedent]
        active = np.all(X[:, cols] == 1, axis=1)
        n_active = int(active.sum())
        if n_active == 0:
            observed_prob = np.nan
            delta = 0.0
        else:
            observed_prob = float(X[active, index[target]].mean())
            delta = float(np.clip((observed_prob - chance) * weight, -1.0, 1.0))
            rule_matrix[cols, index[target]] += delta
        rule_log.append({
            "if": antecedent,
            "then": target,
            "support": n_active,
            "observed_probability": observed_prob,
            "chance": chance,
            "weighted_delta": delta,
        })

    rule_matrix = np.clip(rule_matrix, -1.0, 1.0)
    return {
        "names": names,
        "score": score,
        "conditional_probability": probability,
        "lift": lift,
        "support": support,
        "status": status,
        "chance_p_value": chance_p,
        "programmable_rule_matrix": rule_matrix,
        "rule_log": rule_log,
    }


def print_matrix(title, matrix, names, fmt="{:>9.2f}"):
    print("\n" + title)
    print(" " * 12 + " ".join(f"{name:>9}" for name in names))
    for name, row in zip(names, matrix):
        cells = []
        for value in row:
            if isinstance(value, str):
                cells.append(f"{value[:9]:>9}")
            elif np.isnan(value):
                cells.append(f"{'--':>9}")
            else:
                cells.append(fmt.format(value))
        print(f"{name:>10}  " + " ".join(cells))


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 600
    names = ["A", "B", "C", "D", "E", "F"]
    
    # Allow periodicity to be entered via interactive input
    user_input = input("Enter periodicity (number of steps per cycle, default 50): ").strip()
    try:
        period = float(user_input) if user_input else 50.0
        if period <= 0:
            raise ValueError()
    except ValueError:
        print("Invalid input or non-positive value. Falling back to default period = 50.0")
        period = 50.0

    # Generate baseline probabilities modulated by the user-entered periodicity
    time_steps = np.arange(n)[:, None]
    base_probs = np.array([0.35, 0.30, 0.25, 0.20, 0.18, 0.12])
    # Apply sinusoidal periodic modulation across rows
    modulated_probs = np.clip(base_probs + 0.1 * np.sin(2 * np.pi * time_steps / period), 0.05, 0.95)
    
    X = (rng.random((n, len(names))) < modulated_probs).astype(np.uint8)

    # Plant a degree-2 association: when A and B are both active, C is likely.
    trigger = X[:, 0].astype(bool) & X[:, 1].astype(bool)
    X[trigger, 2] = (rng.random(trigger.sum()) < 0.90).astype(np.uint8)

    # Plant a negative relationship between D and E for demonstration.
    X[X[:, 3].astype(bool), 4] = 0

    programmable_rules = [
        {"if": ["A", "B"], "then": "C", "chance": 0.25, "weight": 1.0},
        {"if": ["D"], "then": "E", "chance": 0.18, "weight": 0.8},
    ]

    out = build_non_association_matrix(
        X, names, rules=programmable_rules, min_support=5,
        min_abs_lift_deviation=0.05, smoothing=1.0, permutations=200
    )
    
    print(f"\nUsing periodicity (cycle length) = {period}")
    print_matrix("Pairwise signed association matrix [-1, +1]", out["score"], names)
    print_matrix("Pairwise status matrix", out["status"], names, fmt="{}")
    print_matrix("Conditional probability matrix P(column | row)",
                 out["conditional_probability"], names)
    print_matrix("Programmable directed rule matrix", out["programmable_rule_matrix"], names)

    print("\nProgrammable rule evaluation:")
    for item in out["rule_log"]:
        print(item)
