"""
Combined Directional Residuals & Frequency-Antagonistic Association Loop
======================================================================
Merges directional n-gram models with a frequency-antagonistic association
matrix. Operates in an interactive loop allowing dynamic text entry and manual
frequency threshold input prior to matrix construction.

Only dependency: NumPy
"""

import math
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple
import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)

BUILTIN_CORPUS = (
    "The process begins with a simple observation of local flows. "
    "As patterns emerge, structural constraints shape the boundary conditions. "
    "Non-isomorphic transformations map states across mismatched dimensions. "
    "Causal ascent drives the system toward higher informational complexity. "
    "Through adiabatic continuation, the generator balances exploration and selective focus."
)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


class NGramModel:
    """Smoothed n-gram model with backoff to shorter contexts."""
    reverse = False

    def __init__(self, order: int = 3, alpha: float = 0.25):
        self.order = max(1, int(order))
        self.alpha = max(1e-9, float(alpha))
        self.counts = [defaultdict(Counter) for _ in range(self.order)]
        self.vocab = Counter()
        self.total_tokens = 0
        self._cache: Dict[tuple, Tuple[Counter, float]] = {}

    def train(self, text: str) -> None:
        self._cache.clear()
        toks = tokenize(text)
        if self.reverse:
            toks = toks[::-1]
        tokens = ["<bos>"] * (self.order - 1) + toks + ["<eos>"]
        for i in range(self.order - 1, len(tokens)):
            target = tokens[i]
            self.vocab[target] += 1
            self.total_tokens += 1
            for ctx_len in range(self.order):
                context = tuple(tokens[i - ctx_len:i]) if ctx_len else ()
                self.counts[ctx_len][context][target] += 1

    def _mix(self, context_tokens: Sequence[str]) -> Tuple[Counter, float]:
        context = list(context_tokens)
        key = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        counter = Counter()
        for ctx_len in range(min(self.order - 1, len(context)), -1, -1):
            k = tuple(context[-ctx_len:]) if ctx_len else ()
            found = self.counts[ctx_len].get(k)
            if not found:
                continue
            weight = 1.0 + 0.7 * ctx_len
            for token, count in found.items():
                counter[token] += weight * count
            if ctx_len >= 1:
                break
        if not counter:
            counter.update(self.vocab)
        result = (counter, sum(counter.values()))
        self._cache[key] = result
        return result

    def logprob(self, context_tokens: Sequence[str], token: str) -> float:
        counter, total = self._mix(context_tokens)
        vocab_size = max(1, len(self.vocab))
        prob = (counter.get(token, 0.0) + self.alpha) / (total + self.alpha * vocab_size)
        return math.log(max(prob, 1e-12))

    def log_unigram(self, token: str) -> float:
        vocab_size = max(1, len(self.vocab))
        total = max(1, self.total_tokens)
        return math.log((self.vocab.get(token, 0) + self.alpha) / (total + self.alpha * vocab_size))


class ReverseNGramModel(NGramModel):
    """The same corpus read right to left."""
    reverse = True


def chain_logprob(model: NGramModel, seq: Sequence[str]) -> float:
    """Chain-rule log-probability of seq."""
    total = model.log_unigram(seq[0])
    for i in range(1, len(seq)):
        total += model.logprob(seq[:i], seq[i])
    return total


def compute_directional_residuals(fwd: NGramModel, bwd: NGramModel, level: int, boundaries: bool):
    rows = []
    for ctx, targets in fwd.counts[level].items():
        if not boundaries and "<bos>" in ctx:
            continue
        for t, n in targets.items():
            if not boundaries and t == "<eos>":
                continue
            seq = list(ctx) + [t]
            f = chain_logprob(fwd, seq)
            b = chain_logprob(bwd, seq[::-1])
            rows.append((tuple(seq), n, f, b, b - f))
    return rows


def build_frequency_antagonistic_matrix(X, feature_names=None, user_frequency_baseline=1.0):
    """
    Builds a frequency-antagonistic matrix incorporating frequency-based
    counts and dynamic status thresholds sensitive to subtle text shifts.
    """
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    names = list(feature_names or [f"F{i}" for i in range(m)])

    score = np.zeros((m, m), dtype=float)
    support = np.zeros((m, m), dtype=float)
    status = np.full((m, m), "balanced", dtype=object)

    rates = X.mean(axis=0)

    for i in range(m):
        for j in range(i + 1, m):
            # Use dot product or minimum intersection over continuous counts for support
            co_occur = np.sum(np.minimum(X[:, i], X[:, j]))
            support[i, j] = support[j, i] = co_occur

            f_i = rates[i]
            f_j = rates[j]

            if f_i + f_j > 0:
                freq_friction = abs(f_i - f_j) / (f_i + f_j + float(user_frequency_baseline))
                co_occurrence_ratio = co_occur / max(1, n)

                antag_val = (co_occurrence_ratio - (f_i * f_j)) * (1.0 + freq_friction)
            else:
                antag_val = 0.0

            score[i, j] = score[j, i] = antag_val

    # Compute dynamic thresholds based on score distribution variance
    non_zero_scores = score[score != 0]
    spread_threshold = max(1e-5, float(np.std(non_zero_scores) * 0.25)) if non_zero_scores.size > 0 else 0.05

    for i in range(m):
        for j in range(i + 1, m):
            both = support[i, j]
            val = score[i, j]

            if both == 0:
                status[i, j] = status[j, i] = "orthogonal"
            elif val > spread_threshold:
                status[i, j] = status[j, i] = "antag_synergy"
            elif val < -spread_threshold:
                status[i, j] = status[j, i] = "antag_friction"
            else:
                status[i, j] = status[j, i] = "neutral"

    return {
        "names": names,
        "score": score,
        "support": support,
        "status": status,
    }


def fmt_seq(seq: Tuple[str, ...]) -> str:
    return " ".join(seq[:-1]) + "  ->  " + seq[-1]


def print_matrix(title, matrix, names, fmt="{:>9.3f}"):
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


def main() -> None:
    print("=" * 75)
    print(" DIRECTIONAL RESIDUALS & FREQUENCY-ANTAGONISTIC LOOP")
    print("=" * 75)
    print("Type text to analyze. Type 'quit' or 'exit' to terminate.\n")

    current_corpus = BUILTIN_CORPUS

    while True:
        try:
            user_input = input("\n[Enter text corpus (or press Enter to use current/default)]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        elif user_input:
            current_corpus = user_input
            print(f"-> Corpus updated ({len(tokenize(current_corpus))} tokens).")

        # 1. Train Forward & Reverse N-Gram Models
        fwd = NGramModel(order=3, alpha=0.25)
        fwd.train(current_corpus)
        bwd = ReverseNGramModel(order=3, alpha=0.25)
        bwd.train(current_corpus)

        rows = compute_directional_residuals(fwd, bwd, level=1, boundaries=False)
        if not rows:
            print("No events to compare in this corpus.")
            continue

        r = np.array([x[4] for x in rows])
        print(f"\n--- DIRECTIONAL RESIDUALS SUMMARY ---")
        print(f"Events: {len(rows)} | Mean |Residual|: {np.abs(r).mean():.3f} | Max |Residual|: {np.abs(r).max():.3f}")

        ordered = sorted(rows, key=lambda x: x[4])
        print("\nTop Negative Residuals (Forward-dominant mass):")
        print(f"  {fmt_seq(ordered[0][0]):<35} n={ordered[0][1]:<2d} res={ordered[0][4]:+.3f}")

        print("\nTop Positive Residuals (Reverse-dominant mass):")
        print(f"  {fmt_seq(ordered[-1][0]):<35} n={ordered[-1][1]:<2d} res={ordered[-1][4]:+.3f}")

        # 2. Prompt user for frequency input BEFORE building the matrix
        print("\n--- MATRIX CONFIGURATION ---")
        freq_input = input("[Enter frequency baseline/threshold value for the matrix, e.g., 1.0]: ").strip()
        try:
            user_frequency_baseline = float(freq_input) if freq_input else 1.0
        except ValueError:
            print("[Invalid number entered; defaulting to 1.0]")
            user_frequency_baseline = 1.0

        # 3. Vocabulary & Frequency-Antagonistic Matrix Construction (Count-accumulating X)
        vocab_items = [token.lower() for token, freq in fwd.vocab.most_common(8) if token not in ("<bos>", "<eos>")]
        if len(vocab_items) < 2:
            vocab_items = list(fwd.vocab.keys())[:5]

        sentences = [tokenize(s) for s in re.split(r'[.!?\n]+', current_corpus) if s.strip()]
        if len(sentences) < 2:
            words = tokenize(current_corpus)
            chunk_size = max(2, len(words) // 4)
            sentences = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

        X = np.zeros((len(sentences), len(vocab_items)), dtype=float)
        token_to_idx = {tok: i for i, tok in enumerate(vocab_items)}
        for s_idx, sent in enumerate(sentences):
            for tok in sent:
                t_lower = tok.lower()
                if t_lower in token_to_idx:
                    X[s_idx, token_to_idx[t_lower]] += 1.0

        if X.shape[0] >= 2 and X.shape[1] >= 2:
            matrix_out = build_frequency_antagonistic_matrix(
                X, feature_names=vocab_items, user_frequency_baseline=user_frequency_baseline
            )
            print_matrix(f"Frequency-Antagonistic Matrix (Baseline={user_frequency_baseline})", matrix_out["score"], vocab_items)
            print_matrix("Antagonistic Status Matrix", matrix_out["status"], vocab_items, fmt="{}")
        else:
            print("\n[Notice: Insufficient text length for full matrix generation.]")


if __name__ == "__main__":
    main()
