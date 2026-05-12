import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss, acf
import os
from datetime import datetime

# --- Configuration Path ---
a_values_file = ''

# --- Check if file exists ---
if not os.path.exists(a_values_file):
    print(f"Error: File does not exist, please check the path: {a_values_file}")
    exit(1)

print(f"Loading file: {a_values_file}")
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Load values ---
print("Loading values...")
a_values = []

with open(a_values_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                try:
                    a_val = float(parts[-1].strip())
                    a_values.append(a_val)
                except ValueError:
                    continue

a_values = np.array(a_values)

if len(a_values) == 0:
    print("Error: No valid values found.")
    exit(1)

print("Loading successful!")
print(f"Data type: {a_values.dtype}")
print(f"Number of samples: {len(a_values)}")

print("\n" + "="*50)
print("="*50)

np.random.seed(42)
a_values_shuffled = a_values.copy()
np.random.shuffle(a_values_shuffled)

# ==================================================
# KPSS inspection (stability inspection)
# ==================================================
print("\n" + "="*50)
print("KPSS inspection results (Null hypothesis: The series is stationary)")
print("="*50)

try:
    # Region='c 'indicates that the inspection level is stable, nlags='auto' selects the lag order automatically
    kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(a_values_shuffled, regression='c', nlags='auto')

    print(f"KPSS statistic: {kpss_stat:.6f}")
    print(f"p-value: {kpss_pval:.6f}")
    print(f"Lags: {kpss_lags}")
    print("Critical values:")
    for key, val in kpss_crit.items():
        print(f"  {key}: {val:.6f}")

    alpha = 0.05
    print("\nConclusion:", end=" ")
    if kpss_pval < alpha:
        print(f"At the {alpha:.0%} significance level, reject the null hypothesis")
        print("  -> The series is non-stationary (trending or unit root present)")
    else:
        print(f"At the {alpha:.0%} significance level, fail to reject the null hypothesis")
        print("  -> The series is stationary")

except Exception as e:
    print(f"KPSS inspection error: {e}")

# ==================================================
# Block Maxima
# ==================================================
print("\n" + "="*50)
print("Block Maxima")
print("="*50)

block_size = 100
num_blocks = len(a_values_shuffled) // block_size

print(f"Shuffled sequence length: {len(a_values_shuffled)}")
print(f"Block size: {block_size}")
print(f"Number of blocks: {num_blocks}")

# Trim the sequence to a length that is divisible by block_size
trimmed_length = num_blocks * block_size
a_values_trimmed = a_values_shuffled[:trimmed_length]

# Reshape the array to form blocks
blocks = a_values_trimmed.reshape(num_blocks, block_size)

# Compute the maximum value of each block
block_maxima = np.max(blocks, axis=1)

# ==================================================
# Berman index calculation (independence test)
# ==================================================
print("\n" + "="*50)
print("Berman index results (for block maxima sequence)")
print("Null hypothesis: The block maxima sequence is independent")
print("="*50)

# Compute autocorrelation function
# nlags is set to a reasonable value, e.g., 20 or 1/4 of block_maxima length
max_lag = min(20, len(block_maxima) // 4)
acf_values = acf(block_maxima, nlags=max_lag, fft=False)

# acf_values[0] is the autocorrelation at lag 0, which is always 1, we need to start from lag 1
acf_lag1_to_m = acf_values[1:]

# Compute Berman index
berman_stat = np.mean(np.abs(acf_lag1_to_m))

print(f"Maximum lag order (m): {max_lag}")
print(f"First {max_lag} lag autocorrelations: {acf_lag1_to_m}")
print(f"Berman index: {berman_stat:.6f}")

threshold = 0.2
print("\nConclusion:", end=" ")
if berman_stat < threshold:
    print(f"Berman index ({berman_stat:.4f}) < {threshold}")
    print("  -> The block maxima sequence can be considered independent (satisfies i.i.d. condition)")
else:
    print(f"Berman index ({berman_stat:.4f}) >= {threshold}")
    print("  -> The block maxima sequence is not independent (autocorrelation present)")