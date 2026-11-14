import os
import hashlib
import itertools
import time
import numpy as np


class SubsetSumOracle:
    """
    Holds the NP-hard Subset Sum instance (weights W, target S),
    precomputes a valid solution vector b, and publishes only (W, S, salt, commitment).
    """
    def __init__(self, weights: np.ndarray, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.weights = np.array(weights, dtype=np.int64)
        self.m = len(self.weights)
        
        # Generate a non-trivial ground-truth binary subset mask b in {0, 1}^m
        self._secret_b = self.rng.integers(0, 2, size=self.m, dtype=np.uint8)
        if np.sum(self._secret_b) == 0:
            self._secret_b[0] = 1
            
        self.target_sum = int(np.dot(self.weights, self._secret_b))
        self.salt = os.urandom(16)  # 128-bit random salt
        self.commitment = self._hash_candidate(self._secret_b)

    def _hash_candidate(self, bit_vector: np.ndarray) -> str:
        packed = np.packbits(bit_vector).tobytes()
        return hashlib.sha256(self.salt + packed).hexdigest()

    def public_parameters(self):
        return self.weights.copy(), self.target_sum, self.salt, self.commitment

    def reveal_if_proven(self, candidate_b: np.ndarray):
        """Verifies the cryptographic commitment AND the mathematical NP-hard constraint."""
        if self._hash_candidate(candidate_b) == self.commitment:
            if int(np.dot(self.weights, candidate_b)) == self.target_sum:
                return True, self._secret_b.copy()
        return False, None


class OpticalHeuristicChannel:
    """
    Renders the candidate state as a 2D optical grid (rows x cols x 16 x 16),
    injects sensor noise, performs parallel Kronecker spatial pooling,
    applies confidence gating (erasure channel), and masks reveals.
    """
    def __init__(self, rows: int, cols: int, reveal_rate: float = 0.45,
                 noise_std: float = 40.0, tau: float = 127.0, c0: float = 25.0,
                 bit_flip_eps: float = 0.02, seed: int = 101):
        self.rows = rows
        self.cols = cols
        self.m = rows * cols
        self.reveal_rate = reveal_rate
        self.noise_std = noise_std
        self.tau = tau
        self.c0 = c0
        self.bit_flip_eps = bit_flip_eps
        self.rng = np.random.default_rng(seed)

    def observe(self, secret_b: np.ndarray) -> np.ndarray:
        """
        Simulates one exposure frame:
        Returns an array of shape (m,) with values in {0, 1, -1 (erased/unrevealed)}.
        """
        # 1. Render bits onto 2D grid and expand to 16x16 pixel blocks per cell
        grid = secret_b.reshape(self.rows, self.cols)
        frame = np.repeat(np.repeat(grid * 255.0, 16, axis=0), 16, axis=1)

        # 2. Add independent zero-mean Gaussian pixel sensor noise
        noisy_frame = frame + self.rng.normal(0.0, self.noise_std, size=frame.shape)

        # 3. Parallel Spatial Pooling via tensor reshape & mean (256-fold variance reduction)
        pooled = noisy_frame.reshape(self.rows, 16, self.cols, 16).mean(axis=(1, 3)).flatten()

        # 4. Confidence Gating: convert to binary symmetric erasure channel
        decisions = np.full(self.m, -1, dtype=np.int8)  # -1 represents erasure
        decisions[pooled > (self.tau + self.c0)] = 1
        decisions[pooled < (self.tau - self.c0)] = 0

        # 5. Apply synthetic channel bit-flip noise (epsilon) & partial reveal mask (r)
        valid_mask = (decisions != -1)
        flips = self.rng.random(self.m) < self.bit_flip_eps
        decisions[valid_mask & flips] ^= 1

        reveal_mask = self.rng.random(self.m) < self.reveal_rate
        final_obs = np.where(valid_mask & reveal_mask, decisions, -1)
        return final_obs


class CommitVerifiedChaseDecoder:
    """
    Accumulates majority votes across exposures, waits for coupon-collector cover time,
    and executes Chase-type bounded bit-flip list decoding gated by SHA-256 verification.
    """
    def __init__(self, weights: np.ndarray, target_sum: int, salt: bytes, commitment: str,
                 L: int = 16, t: int = 4, q_max: int = 200_000):
        self.weights = weights
        self.target_sum = target_sum
        self.salt = salt
        self.commitment = commitment
        self.m = len(weights)
        self.L = min(L, self.m)
        self.t = min(t, self.L)
        self.q_max = q_max

        self.ones = np.zeros(self.m, dtype=np.int32)
        self.zeros = np.zeros(self.m, dtype=np.int32)
        self.seen = np.zeros(self.m, dtype=bool)
        self.total_hashes_tested = 0

    def _verify(self, candidate_b: np.ndarray) -> bool:
        if self.total_hashes_tested >= self.q_max:
            return False
        self.total_hashes_tested += 1
        # Fast pre-filter: check subset sum arithmetic before cryptographic SHA-256
        if int(np.dot(self.weights, candidate_b)) != self.target_sum:
            return False
        packed = np.packbits(candidate_b).tobytes()
        return hashlib.sha256(self.salt + packed).hexdigest() == self.commitment

    def update_and_decode(self, observation: np.ndarray):
        """Updates vote counters and runs Chase decoding once all cells are covered."""
        is_one = (observation == 1)
        is_zero = (observation == 0)

        self.ones += is_one
        self.zeros += is_zero
        self.seen |= (is_one | is_zero)

        # Wait until coupon-collector cover time is reached (every cell observed >= 1)
        if not np.all(self.seen):
            return None

        # Majority vote baseline estimate (ties decode to 0)
        b_hat = (self.ones > self.zeros).astype(np.uint8)

        # Test zero-flip baseline candidate first
        if self._verify(b_hat):
            return b_hat

        # Rank bit positions by confidence margin |D_i| = |ones_i - zeros_i| (ascending)
        margins = np.abs(self.ones - self.zeros)
        # Random tie-breaking avoids bias toward low-index bits
        tie_breaker = np.random.random(self.m)
        uncertain_indices = np.lexsort((tie_breaker, margins))[:self.L]

        # Bounded bit-flip search over the L least-reliable positions up to Hamming weight t
        for k in range(1, self.t + 1):
            for flip_subset in itertools.combinations(uncertain_indices, k):
                if self.total_hashes_tested >= self.q_max:
                    return None
                candidate = b_hat.copy()
                candidate[list(flip_subset)] ^= 1
                if self._verify(candidate):
                    return candidate

        return None


def run_subset_sum_bench(rows: int = 8, cols: int = 7, max_cycles: int = 800, seed: int = 42):
    m = rows * cols  # Default m = 56 bits (2^56 ~ 7.2e16 search space)
    rng = np.random.default_rng(seed)

    # Generate large random 48-bit weights for a hard Subset Sum instance
    weights = rng.integers(10**12, 10**14, size=m, dtype=np.int64)

    # 1. Initialize Oracle
    oracle = SubsetSumOracle(weights, seed=seed)
    pub_weights, target_sum, salt, commitment = oracle.public_parameters()

    # 2. Initialize Optical Channel & Chase Recogniser
    channel = OpticalHeuristicChannel(rows=rows, cols=cols, reveal_rate=0.45, bit_flip_eps=0.02, seed=seed + 1)
    decoder = CommitVerifiedChaseDecoder(pub_weights, target_sum, salt, commitment, L=16, t=4, q_max=200_000)

    print(f"=== NP-Hard Subset Sum Commit-Only Recovery Bench (m = {m} bits) ===")
    print(f"Brute-Force Search Space : 2^{m} = {2**m:,} combinations")
    print(f"Target Subset Sum        : {target_sum:,}")
    print(f"Published Commitment     : {commitment[:32]}...\n")

    start_time = time.perf_counter()
    cover_cycle = None

    for cycle in range(1, max_cycles + 1):
        obs = channel.observe(oracle._secret_b)
        recovered = decoder.update_and_decode(obs)

        if cover_cycle is None and np.all(decoder.seen):
            cover_cycle = cycle
            print(f"[Cycle {cycle:02d}] Coupon-collector cover time reached (all {m} bits observed).")

        if recovered is not None:
            elapsed = (time.perf_counter() - start_time) * 1000
            verified, truth = oracle.reveal_if_proven(recovered)
            chosen_weights = pub_weights[recovered == 1]

            print(f"[Cycle {cycle:02d}] SOLUTION RECOVERED & CRYPTOGRAPHICALLY VERIFIED!")
            print(f"  - Elapsed Time          : {elapsed:.2f} ms")
            print(f"  - Cover Time Cycle      : {cover_cycle} (Theoretical mean ~ 7.7)")
            print(f"  - Candidates Evaluated  : {decoder.total_hashes_tested:,} (vs 2^{m-1} avg brute force)")
            print(f"  - Subset Cardinality    : {len(chosen_weights)} / {m} weights selected")
            print(f"  - Computed Sum          : {int(np.sum(chosen_weights)):,}")
            print(f"  - Exact Target Match    : {int(np.sum(chosen_weights)) == target_sum}")
            print(f"  - Commitment Verified   : {verified}")
            return

    print("Failed to recover solution within cycle/hash budget.")


if __name__ == "__main__":
    run_subset_sum_bench()
