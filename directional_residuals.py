"""
Combined Directional Residuals & Frequency-Antagonistic Association Loop
===================================================================================
Updated with adaptive support thresholds to prevent 'insufficient' labels on shorter inputs.
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
    reverse = True


def chain_logprob(model: NGramModel, seq: Sequence[str]) -> float:
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


def build_frequency_antagonistic_matrix(
    X,
    feature_names=None,
    min_support=1,  # Lowered default to 1 to accommodate short texts
    min_abs_lift_deviation=0.01,
    smoothing=1.0,
):
    X = np.asarray(X, dtype=np.uint8)
    n, m = X.shape
    names = list(feature_names or [f"F{i}" for i in range(m)])

    score = np.zeros((m, m), dtype=float)
    probability = np.full((m, m), np.nan, dtype=float)
    support = np.zeros((m, m), dtype=int)
    status = np.full((m, m), "diagonal", dtype=object)

    rates = X.mean(axis=0)
    antagonistic_weights = 1.0 / (1.0 + np.log(1.0 + rates * n))

    for i in range(m):
        for j in range(i + 1, m):
            both = int(np.sum(X[:, i] & X[:, j]))
            count_i = int(X[:, i].sum())
            count_j = int(X[:, j].sum())
            support[i, j] = support[j, i] = both

            p_j_given_i = (both + smoothing * rates[j]) / (count_i + smoothing) if count_i + smoothing else rates[j]
            probability[i, j] = p_j_given_i

            expected = rates[i] * rates[j]
            observed = both / n
            denom = min(expected, rates[i] * (1 - rates[j]), rates[j] * (1 - rates[i]),
                        (1 - rates[i]) * (1 - rates[j]))
            raw = (observed - expected) / denom if denom > 0 else 0.0
            raw = float(np.clip(raw, -1.0, 1.0))

            antagonism_factor = (antagonistic_weights[i] + antagonistic_weights[j]) / 2.0
            adjusted_score = raw * antagonism_factor

            score[i, j] = score[j, i] = adjusted_score

            effect = abs(p_j_given_i - rates[j])
            if both < min_support:
                label = "insufficient"
            elif effect < min_abs_lift_deviation:
                label = "non_assoc"
            else:
                label = "antag_pos" if adjusted_score > 0 else "antag_neg"
            status[i, j] = status[j, i] = label

    return {
        "names": names,
        "score": score,
        "conditional_probability": probability,
        "support": support,
        "status": status,
        "antagonistic_weights": antagonistic_weights,
    }


def fmt_seq(seq: Tuple[str, ...]) -> str:
    return " ".join(seq[:-1]) + "  ->  " + seq[-1]


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


def main():
    print("=" * 75)
    print(" DIRECTIONAL RESIDUALS & FREQUENCY-ANTAGONISTIC ASSOCIATION LOOP")
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

        # Build vocabulary items (take top 8 most frequent words)
        vocab_items = [token.lower() for token, freq in fwd.vocab.most_common(8) if token not in ("<bos>", "<eos>")]
        if len(vocab_items) < 2:
            vocab_items = list(fwd.vocab.keys())[:5]

        # Break text into sliding windows / sentences
        sentences = [tokenize(s) for s in re.split(r'[.!?\n]+', current_corpus) if s.strip()]
        if len(sentences) < 2:
            # If text is a single continuous block, chunk it into synthetic windows of 5 words
            words = tokenize(current_corpus)
            chunk_size = max(3, len(words) // 4)
            sentences = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

        X = np.zeros((len(sentences), len(vocab_items)), dtype=np.uint8)
        token_to_idx = {tok: i for i, tok in enumerate(vocab_items)}
        for s_idx, sent in enumerate(sentences):
            sent_set = {t.lower() for t in sent}
            for tok in sent_set:
                if tok in token_to_idx:
                    X[s_idx, token_to_idx[tok]] = 1

        if X.shape[0] >= 2 and X.shape[1] >= 2:
            matrix_out = build_frequency_antagonistic_matrix(X, feature_names=vocab_items, min_support=1, min_abs_lift_deviation=0.001)
            print_matrix("Frequency-Antagonistic Association Matrix [-1, +1]", matrix_out["score"], vocab_items)
            print_matrix("Antagonistic Status Matrix", matrix_out["status"], vocab_items, fmt="{}")
        else:
            print("\n[Notice: Insufficient text variability to build association matrix.]")


if __name__ == "__main__":
    main()
