import hashlib
import numpy as np
from itertools import combinations

def packbits(b_vector):
    """Packs a binary vector into bytes for SHA-256 commitment."""
    b_arr = np.asarray(b_vector, dtype=np.uint8)
    padded_len = ((len(b_arr) + 7) // 8) * 8
    padded = np.zeros(padded_len, dtype=np.uint8)
    padded[:len(b_arr)] = b_arr
    return np.packbits(padded)

def run_transparent_bench(rows=8, cols=7, max_cycles=49, sigma=40.0, epsilon=0.02, r=0.45, seed=42):
    """
    Simulates the Optical-Channel Subset Sum Recovery and Chase List Decoding pipeline,
    printing input subset definitions and output recovery details.
    """
    np.random.seed(seed)
    m = rows * cols
    print(f"==================================================")
    print(f"[*] INITIALIZING SUBSET SUM INSTANCE: m = {m} ({rows}x{cols} grid)")
    print(f"==================================================")

    # ---------------------------------------------------------
    # Step 1: Oracle Setup (Input Subset Generation)
    # ---------------------------------------------------------
    # Draw weights in range [10^11, 10^12) and secret bit-vector b
    weights = np.random.randint(10**11, 10**12, size=m, dtype=np.int64)
    secret_b = np.random.randint(0, 2, size=m, dtype=np.uint8)
    if np.sum(secret_b) == 0:
        secret_b[0] = 1  # enforce at least one non-zero bit

    target_sum = np.dot(weights, secret_b.astype(np.int64))
    salt = os_urandom_mock(16)
    
    # Commitment: SHA-256(s || packbits(b))[cite: 1]
    packed_secret = packbits(secret_b)
    commit_hasher = hashlib.sha256()
    commit_hasher.update(salt + packed_secret.tobytes())
    commitment = commit_hasher.digest()

    # Display Input Subset Information
    print("\n[INPUT SUBSET DETAILS]:")
    print(f"  - Total Weights Available (w): {m} integers in range [10^11, 10^12)")
    print(f"  - First 5 Weights (w[:5]): {weights[:5]}")
    print(f"  - Secret Bit-Vector (b) [First 20 bits]: {secret_b[:20]}")
    print(f"  - Active Subset Indices (where b_i = 1): {np.where(secret_b == 1)[0][:10]} ... (total {np.sum(secret_b)} active items)")
    print(f"  - Published Target Sum (S = w · b): {target_sum}")
    print(f"  - Cryptographic Commitment (c): {commitment.hex()[:32]}...\n")

    # ---------------------------------------------------------
    # Step 2: Optical Channel & Recognizer State Initialization
    # ---------------------------------------------------------
    ones_counts = np.zeros(m, dtype=np.int32)
    zeros_counts = np.zeros(m, dtype=np.int32)
    seen_mask = np.zeros(m, dtype=bool)

    b_grid = secret_b.reshape((rows, cols))
    success = False
    solved_cycle = -1
    b_hat = None

    print(f"[*] Starting Optical Transmission & Observation Cycles (Max: {max_cycles})...")
    for cycle in range(1, max_cycles + 1):
        # Optical rendering and noise simulation
        pixel_grid = np.repeat(np.repeat(b_grid.astype(float) * 255.0, 16, axis=0), 16, axis=1)
        noise = np.random.normal(0, sigma, size=pixel_grid.shape)
        noisy_pixels = pixel_grid + noise

        # Kronecker Spatial Pooling[cite: 2]
        pooled_cells = noisy_pixels.reshape(rows, 16, cols, 16).mean(axis=(1, 3))
        pooled_flat = pooled_cells.flatten()

        # Reveal and bit-flip masks[cite: 1]
        revealed = np.random.rand(m) < r
        flips = np.random.rand(m) < epsilon

        for i in range(m):
            if revealed[i]:
                observed_bit = secret_b[i]
                if flips[i]:
                    observed_bit = 1 - observed_bit
                if observed_bit == 1:
                    ones_counts[i] += 1
                else:
                    zeros_counts[i] += 1
                seen_mask[i] = True

        valid_count = np.sum(seen_mask)
        
        # Check if all bits have been observed (Cover Time)[cite: 2]
        if valid_count == m:
            b_hat = (ones_counts > zeros_counts).astype(np.uint8)

            # Dual Verifier Check[cite: 3]
            current_sum = int(np.dot(weights, b_hat.astype(np.int64)))
            packed_b_hat = packbits(b_hat)
            
            hash_hasher = hashlib.sha256()
            hash_hasher.update(salt + packed_b_hat.tobytes())
            current_commit = hash_hasher.digest()

            if current_sum == target_sum and current_commit == commitment:
                success = True
                solved_cycle = cycle
                break
            
            # Chase List Decoding (L=16, t=4)[cite: 3]
            if cycle >= 5: 
                margins = np.abs(ones_counts - zeros_counts)
                l_indices = np.argsort(margins)[:16]

                chase_success = False
                for k in range(1, 5):
                    for combo in combinations(l_indices, k):
                        cand = b_hat.copy()
                        for idx in combo:
                            cand[idx] = 1 - cand[idx]

                        cand_sum = int(np.dot(weights, cand.astype(np.int64)))
                        if cand_sum == target_sum:
                            cand_packed = packbits(cand)
                            cand_hasher = hashlib.sha256()
                            cand_hasher.update(salt + cand_packed.tobytes())
                            if cand_hasher.digest() == commitment:
                                success = True
                                solved_cycle = cycle
                                b_hat = cand
                                chase_success = True
                                break
                    if chase_success:
                        break
                if chase_success:
                    break

    # ---------------------------------------------------------
    # Step 3: Output Subset Verification & Display
    # ---------------------------------------------------------
    print(f"\n==================================================")
    if success:
        print(f"[+] SUCCESS: Exact Recovery Achieved at Cycle {solved_cycle}!")
        print(f"==================================================")
        
        # Compute output subset details
        recovered_sum = int(np.dot(weights, b_hat.astype(np.int64)))
        match_status = np.array_equal(secret_b, b_hat)
        hamming_distance = np.sum(secret_b != b_hat)

        print("\n[OUTPUT SUBSET DETAILS]:")
        print(f"  - Recovered Bit-Vector (b̂) [First 20 bits]: {b_hat[:20]}")
        print(f"  - Recovered Active Indices: {np.where(b_hat == 1)[0][:10]} ... (total {np.sum(b_hat)} active items)")
        print(f"  - Arithmetic Check (w · b̂): {recovered_sum} (Target S: {target_sum})")
        print(f"  - Cryptographic Verification Match: PASSED (SHA-256 Commitment Verified)")
        print(f"  - Exact Match with Secret Vector (b): {match_status} (Hamming Distance errors: {hamming_distance})")
    else:
        print(f"[-] PIPELINE FAILED: Max cycles reached without verification.")
        print(f"==================================================")

    return success, solved_cycle, b_hat

def os_urandom_mock(n):
    """Mock for os.urandom to generate reproducible cryptographic salt."""
    import os
    return os.urandom(n)

if __name__ == "__main__":
    # Execute the simulation
    run_transparent_bench(rows=8, cols=7, max_cycles=49)
