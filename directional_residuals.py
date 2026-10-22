"""
Directional residuals of a forward vs time-reversed n-gram model
================================================================

Mapping only. For an observed n-gram (c -> t at level 1, a b -> t at level 2) the model can
score the SAME joint event two ways, by the chain rule:

  forward  : log U(c) + log Pf(t | c)              (level 2: log U(a) + log Pf(b|a) + log Pf(t|a,b))
  backward : log U(t) + log Pb(c | t)              (level 2: log U(t) + log Pb(b|t) + log Pb(a|b,t))

  residual = backward - forward

For raw counts the two factorizations are identical (both equal n(event)/N), so the residual
is exactly 0: a counting model has no intrinsic arrow of time. Whatever residual remains is
what the estimator adds: additive smoothing, backoff weighting, and boundary tokens. That makes
the residual a map of where the model's forward and backward views disagree, and it should
shrink as evidence grows. The table by observation count tests exactly that.

  positive residual : the reversed reading gives the event more mass than the forward reading
  negative residual : the forward reading gives it more mass

Boundary tokens (<bos>, <eos>) are excluded by default because their asymmetry is structural.

Examples:
  python directional_residuals.py corpus.txt
  python directional_residuals.py corpus.txt --level 2 --order 3 --top 12
  python directional_residuals.py corpus.txt --alpha 0.01        # less smoothing -> smaller residual
  python directional_residuals.py corpus.txt --boundaries

Only dependency: NumPy
"""

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
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
    """Chain-rule log-probability of seq: log U(seq[0]) + sum log P(seq[i] | seq[:i])."""
    total = model.log_unigram(seq[0])
    for i in range(1, len(seq)):
        total += model.logprob(seq[:i], seq[i])
    return total


def directional_residuals(fwd: NGramModel, bwd: NGramModel, level: int, boundaries: bool):
    """Rows of (context+target tuple, observation count, forward, backward, residual)."""
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


def fmt(seq: Tuple[str, ...]) -> str:
    return " ".join(seq[:-1]) + "  ->  " + seq[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Map the directional residual between a forward and a time-reversed n-gram model.")
    parser.add_argument("corpus", default="xaa", nargs="?", help="UTF-8 text corpus file; uses a small built-in corpus if the default file is absent.")
    parser.add_argument("--level", type=int, choices=[1, 2], default=1, help="Context length of the events compared (1 = bigrams, 2 = trigrams).")
    parser.add_argument("--order", type=int, default=4, help="N-gram order of both models (must exceed --level).")
    parser.add_argument("--alpha", type=float, default=0.25, help="Additive smoothing value.")
    parser.add_argument("--top", type=int, default=8, help="Rows shown per list.")
    parser.add_argument("--boundaries", action="store_true", help="Include events touching <bos>/<eos>.")
    args = parser.parse_args()

    if args.order <= args.level:
        parser.error("--order must be greater than --level")

    corpus_path = Path(args.corpus)
    if corpus_path.is_file():
        try:
            corpus_text = corpus_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            parser.error(f"Could not decode corpus as UTF-8: {exc}")
    elif args.corpus == parser.get_default("corpus"):
        print(f"[corpus '{args.corpus}' not found; using the built-in corpus]", file=sys.stderr)
        corpus_text = BUILTIN_CORPUS
    else:
        parser.error(f"Corpus file not found: {corpus_path}")

    if not tokenize(corpus_text):
        parser.error("Corpus contains no recognizable tokens.")

    fwd = NGramModel(order=args.order, alpha=args.alpha)
    fwd.train(corpus_text)
    bwd = ReverseNGramModel(order=args.order, alpha=args.alpha)
    bwd.train(corpus_text)

    rows = directional_residuals(fwd, bwd, args.level, args.boundaries)
    if not rows:
        print("No events to compare.")
        return

    r = np.array([x[4] for x in rows])
    n = np.array([x[1] for x in rows])
    print(f"DIRECTIONAL RESIDUALS  level={args.level} ({'bigram' if args.level == 1 else 'trigram'} events)  "
          f"order={args.order}  alpha={args.alpha}")
    print("  residual = backward chain log-prob - forward chain log-prob  (0 for raw counts)\n")
    print(f"  events: {len(rows)}   mean|residual|={np.abs(r).mean():.3f}   max|residual|={np.abs(r).max():.3f}   "
          f"mean signed={r.mean():+.3f}")

    print("\n  by observation count (does the residual shrink as evidence grows?):")
    print("    count   events   mean|residual|")
    for label, mask in (("1", n == 1), ("2", n == 2), ("3-4", (n >= 3) & (n <= 4)), ("5+", n >= 5)):
        if mask.any():
            print(f"    {label:>5} {int(mask.sum()):8d}   {np.abs(r[mask]).mean():.3f}")

    ordered = sorted(rows, key=lambda x: x[4])
    print("\n  negative (forward reading gives the event more mass):")
    for seq, cnt, f, b, res in ordered[:args.top]:
        print(f"    {fmt(seq):<40} n={cnt:<3d} residual={res:+.3f}")
    print("\n  positive (reversed reading gives the event more mass):")
    for seq, cnt, f, b, res in ordered[::-1][:args.top]:
        print(f"    {fmt(seq):<40} n={cnt:<3d} residual={res:+.3f}")


if __name__ == "__main__":
    main()
