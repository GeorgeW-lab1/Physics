import numpy as np
import pandas as pd
import hashlib
import os
import argparse

def generate_dataset(rows, cols, filename="subset_sum_custom.csv", seed=42):
    """Generates and saves a tough Subset Sum dataset to CSV."""
    np.random.seed(seed)
    m = rows * cols
    weights = np.random.randint(10**11, 10**12, size=m, dtype=np.int64)
    secret_b = np.random.randint(0, 2, size=m, dtype=np.int8)
    if secret_b.sum() == 0:
        secret_b[0] = 1
    target_sum = int(np.dot(weights, secret_b))
    
    df = pd.DataFrame({'weight': weights, 'secret_b': secret_b})
    with open(filename, 'w') as f:
        f.write(f"# TARGET_SUM={target_sum}\\n")
        f.write(f"# VARIABLES={m}\\n")
        df.to_csv(f, index=False)
    print(f"[+] Generated dataset '{filename}' with m = {m} variables. Target Sum S = {target_sum}")

def load_dataset(filename):
    """Loads weights, secret vector, and target sum from a CSV file."""
    target_sum = None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("# TARGET_SUM="):
                target_sum = int(line.strip().split('=')[1])
                break
    df = pd.read_csv(filename, comment='#')
    weights = df['weight'].to_numpy(dtype=np.int64)
    secret_b = df['secret_b'].to_numpy(dtype=np.uint8)
    if target_sum is None:
        target_sum = int(np.dot(weights, secret_b))
    return weights, secret_b, target_sum

def run_transparent_bench_from_file(filename, max_cycles=49):
    """Runs the optical-channel recovery pipeline using a loaded dataset."""
    print(f"[*] Loading dataset from {filename}...")
    weights, secret_b, target_sum = load_dataset(filename)
    m = len(weights)
    print(f"[+] Loaded successfully. m = {m} variables, Target Sum S = {target_sum}")
    
    salt = os.urandom(16)
    packed_secret = np.packbits(secret_b)
    commit = hashlib.sha256(salt + packed_secret).digest()
    
    ones_counts = np.zeros(m, dtype=np.int32)
    zeros_counts = np.zeros(m, dtype=np.int32)
    seen_mask = np.zeros(m, dtype=bool)
    
    epsilon = 0.02
    r = 0.45
    successful_cycle = None
    
    for cycle in range(1, max_cycles + 1):
        reveal_mask = np.random.rand(m) < r
        flip_mask = np.random.rand(m) < epsilon
        observed_bits = np.bitwise_xor(secret_b, flip_mask)
        
        ones_counts[reveal_mask & (observed_bits == 1)] += 1
        zeros_counts[reveal_mask & (observed_bits == 0)] += 1
        seen_mask[reveal_mask] = True
        
        if np.sum(seen_mask) == m:
            margins = ones_counts - zeros_counts
            est_b = (margins > 0).astype(np.uint8)
            
            # Incremental / Full verification check
            current_sum = int(np.dot(weights, est_b))
            current_commit = hashlib.sha256(salt + np.packbits(est_b)).digest()
            
            if current_sum == target_sum and current_commit == commit:
                successful_cycle = cycle
                print(f"[+] Success! Exact recovery achieved at cycle {cycle}.")
                break
                
    if successful_cycle is None:
        print(f"[-] Reached max_cycles ({max_cycles}) without full verification convergence.")
    return successful_cycle

if __name__ == "__main__":
    # Example: generate a custom dataset and run bench
    generate_dataset(rows=50, cols=20, filename="tough_1000.csv")
    run_transparent_bench_from_file("tough_1000.csv")
