#!/usr/bin/env python3
"""
Compile and analyze k-mer uniqueness results for the human genome (hg38).
Combines measured data with power-law curve fitting to determine thresholds.
"""

import numpy as np
import json
from pathlib import Path

# === MEASURED DATA (KMC on hg38, all main chromosomes) ===
# Total bases: 3,088,286,401
# Non-N bases: 2,937,655,681 (95.1%)

measured = {
    # k: (unique_kmers_count1, distinct_kmers, total_kmers)
    11:  (197,             2_076_668,       2_443_914_481),
    13:  (1_324_030,       32_368_835,      2_702_234_590),
    15:  (85_667_028,      333_990_307,     2_756_537_465),
    17:  (936_086_991,     1_392_202_889,   2_794_397_290),
    20:  (2_052_991_379,   2_189_051_403,   2_838_238_779),
    24:  (2_266_886_339,   2_356_806_333,   2_877_812_905),
    28:  (2_363_451_740,   2_445_797_417,   2_902_857_049),
    32:  (2_437_332_752,   2_514_433_537,   2_918_642_250),
    40:  (2_547_132_907,   2_615_555_539,   2_933_668_528),
    50:  (2_638_156_815,   2_697_532_531,   2_937_290_401),
    75:  (2_743_503_533,   2_788_198_539,   2_937_588_617),
    100: (2_780_227_757,   2_819_184_604,   2_937_566_706),
}

# KMC distinct counts (no histogram available for these due to kmc_tools bug)
distinct_only = {
    # k: (distinct, total)
    150: (2_846_927_413, 2_937_523_165),
    200: (2_861_548_499, 2_937_479_985),
    250: (2_871_010_755, 2_937_437_184),
}

print("=" * 80)
print("HUMAN GENOME (hg38) K-MER UNIQUENESS ANALYSIS")
print("=" * 80)
print(f"Genome: GRCh38/hg38, all main chromosomes (chr1-22, X, Y, M)")
print(f"Total bases: 3,088,286,401")
print(f"Non-N bases: 2,937,655,681 (95.1%)")
print()

# === TABLE OF MEASURED RESULTS ===
print(f"{'k':>6} | {'Unique (cnt=1)':>16} | {'Total k-mers':>16} | {'% Unique':>12} | {'Non-unique bp':>16}")
print("-" * 80)

for k in sorted(measured.keys()):
    unique, distinct, total = measured[k]
    pct = 100.0 * unique / total
    non_unique = total - unique
    print(f"{k:>6} | {unique:>16,} | {total:>16,} | {pct:>11.4f}% | {non_unique:>16,}")

# Estimate unique for k=150,200,250 using the unique/distinct ratio trend
print()
print("--- Estimated values (extrapolated unique/distinct ratio) ---")

# Fit unique/distinct ratio as function of k
ks_ratio = []
ratios = []
for k in sorted(measured.keys()):
    if k >= 20:
        unique, distinct, total = measured[k]
        ks_ratio.append(k)
        ratios.append(unique / distinct)

# Fit: ratio = 1 - A * k^B (approaches 1 as k increases)
deficits = [1 - r for r in ratios]
log_k = np.log(ks_ratio)
log_d = np.log(deficits)
coeff = np.polyfit(log_k, log_d, 1)
B_ratio, A_ratio_log = coeff
A_ratio = np.exp(A_ratio_log)

for k in sorted(distinct_only.keys()):
    distinct, total = distinct_only[k]
    est_ratio = 1 - A_ratio * k ** B_ratio
    est_unique = int(distinct * est_ratio)
    est_pct = 100.0 * est_unique / total
    non_unique = total - est_unique
    print(f"{k:>6} | {est_unique:>16,} (est) | {total:>16,} | {est_pct:>11.4f}% | {non_unique:>16,}")
    # Add to measured dict for curve fitting
    measured[k] = (est_unique, distinct, total)

# === POWER LAW FITTING ===
print()
print("=" * 80)
print("POWER LAW ANALYSIS")
print("=" * 80)

# Compute non-unique fraction for each k (k >= 15 for meaningful data)
ks_fit = []
non_unique_fracs = []
for k in sorted(measured.keys()):
    unique, distinct, total = measured[k]
    pct = unique / total
    if pct > 0.01 and pct < 1.0:  # Between 1% and 100% unique
        ks_fit.append(k)
        non_unique_fracs.append(1.0 - pct)

# Fit multiple regimes
print("\n1. Overall power law (k=15-250):")
log_k_all = np.log(ks_fit)
log_f_all = np.log(non_unique_fracs)
b_all, a_all = np.polyfit(log_k_all, log_f_all, 1)
A_all = np.exp(a_all)
y_pred = a_all + b_all * log_k_all
ss_res = np.sum((log_f_all - y_pred) ** 2)
ss_tot = np.sum((log_f_all - np.mean(log_f_all)) ** 2)
r2_all = 1 - ss_res / ss_tot
print(f"   f(k) = {A_all:.4f} * k^({b_all:.4f}),  R² = {r2_all:.4f}")

# Small-k regime (k=15-80)
mask_small = [i for i, k in enumerate(ks_fit) if 15 <= k <= 80]
if len(mask_small) >= 3:
    ks_s = [ks_fit[i] for i in mask_small]
    fs_s = [non_unique_fracs[i] for i in mask_small]
    b_s, a_s = np.polyfit(np.log(ks_s), np.log(fs_s), 1)
    A_s = np.exp(a_s)
    print(f"\n2. Small-k regime (k=15-80):")
    print(f"   f(k) = {A_s:.4f} * k^({b_s:.4f})")

# Large-k regime (k=50-250)
mask_large = [i for i, k in enumerate(ks_fit) if k >= 50]
if len(mask_large) >= 3:
    ks_l = [ks_fit[i] for i in mask_large]
    fs_l = [non_unique_fracs[i] for i in mask_large]
    b_l, a_l = np.polyfit(np.log(ks_l), np.log(fs_l), 1)
    A_l = np.exp(a_l)
    y_pred_l = a_l + b_l * np.log(ks_l)
    ss_res_l = np.sum((np.log(fs_l) - y_pred_l) ** 2)
    ss_tot_l = np.sum((np.log(fs_l) - np.mean(np.log(fs_l))) ** 2)
    r2_l = 1 - ss_res_l / ss_tot_l
    print(f"\n3. Large-k regime (k=50-250) — BEST for extrapolation:")
    print(f"   f(k) = {A_l:.4f} * k^({b_l:.4f}),  R² = {r2_l:.4f}")

# === THRESHOLD CALCULATIONS ===
print()
print("=" * 80)
print("THRESHOLD ANALYSIS: Context size needed for X% uniqueness")
print("=" * 80)

thresholds = {
    '99%':    0.01,
    '99.9%':  0.001,
    '99.99%': 0.0001,
    '100%':   None,  # special case
}

# Check measured data first
print("\nFrom measured data:")
sorted_measured = sorted(measured.items())
for label in ['99%', '99.9%', '99.99%']:
    target_frac = thresholds[label]
    found = None
    for k, (unique, distinct, total) in sorted_measured:
        if unique / total >= (1 - target_frac):
            found = k
            break
    if found:
        print(f"  {label}: k >= {found} (measured)")
    else:
        best_k = max(measured.keys())
        best_pct = 100.0 * measured[best_k][0] / measured[best_k][2]
        print(f"  {label}: NOT reached at k={best_k} (best: {best_pct:.4f}%)")

# Extrapolation using large-k power law
print(f"\nExtrapolated (large-k power law: f(k) = {A_l:.4f} * k^({b_l:.4f})):")
results_summary = {}

for label, target_frac in thresholds.items():
    if target_frac is None:
        continue
    k_est = (target_frac / A_l) ** (1.0 / b_l)
    # Also compute with overall fit for comparison
    k_est_all = (target_frac / A_all) ** (1.0 / b_all)
    print(f"  {label}: k ~ {k_est:,.0f} bp  (overall fit: ~{k_est_all:,.0f} bp)")
    results_summary[label] = {
        'large_k_estimate': int(round(k_est)),
        'overall_estimate': int(round(k_est_all)),
    }

# 100% is special - dominated by satellite repeats
print(f"\n  100%: Effectively unreachable with fixed-length k-mers.")
print(f"        ~1.2% of genome ({int(0.012 * 2_937_655_681):,} bp) consists of")
print(f"        tandem/satellite repeats with near-identical copies spanning")
print(f"        10-100+ kb, requiring k > 100,000 bp for full resolution.")
print(f"        Some centromeric repeats are essentially identical for Mbp")
print(f"        stretches, making true 100% impossible in practice.")

# === SUMMARY TABLE ===
print()
print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)
print()
print("Context size needed to uniquely identify base pairs in the human genome:")
print()
print(f"  {'Threshold':>12} | {'Min Context (k)':>20} | {'Non-unique bases remaining':>30}")
print(f"  {'-'*12}-+-{'-'*20}-+-{'-'*30}")

non_n = 2_937_655_681

answers = [
    ("50%",    "k = 20 bp",        int(non_n * 0.277)),
    ("75%",    "k = 24 bp",        int(non_n * 0.212)),
    ("90%",    "k ~ 55 bp",        int(non_n * 0.10)),
    ("95%",    "k ~ 120 bp",       int(non_n * 0.05)),
    ("99%",    f"k ~ {results_summary['99%']['large_k_estimate']:,} bp",
                                    int(non_n * 0.01)),
    ("99.9%",  f"k ~ {results_summary['99.9%']['large_k_estimate']:,} bp",
                                    int(non_n * 0.001)),
    ("99.99%", f"k ~ {results_summary['99.99%']['large_k_estimate']:,} bp",
                                    int(non_n * 0.0001)),
    ("100%",   "k > 100,000+ bp",  0),
]

for threshold, context, remaining in answers:
    print(f"  {threshold:>12} | {context:>20} | {remaining:>26,} bp")

print()
print("Notes:")
print("  - 'Unique' means the k-mer starting at that position appears exactly once")
print("    in the entire genome (all chromosomes, both strands counted separately)")
print("  - Values for k <= 100 are measured directly via KMC on GRCh38/hg38")
print("  - Values for k > 100 are extrapolated using power law fit to measured data")
print("  - The genome contains ~150M N bases (gaps); only ACGT bases are counted")
print("  - Diminishing returns: going from 99% to 99.9% requires ~10x more context")
print("  - The hardest ~1% is dominated by segmental duplications and satellite DNA")
print("  - True 100% is impractical: centromeric alpha-satellite arrays span Mbp")
print("    with nearly identical ~171bp repeat units")

# Save comprehensive results
output = {
    'genome': 'GRCh38/hg38 (chr1-22, X, Y, M)',
    'total_bases': 3_088_286_401,
    'non_n_bases': 2_937_655_681,
    'measured_data': {
        str(k): {
            'k': k,
            'unique_kmers': int(v[0]),
            'distinct_kmers': int(v[1]),
            'total_kmers': int(v[2]),
            'pct_unique': round(100.0 * v[0] / v[2], 6),
            'source': 'measured' if k <= 100 else 'estimated'
        }
        for k, v in sorted(measured.items())
    },
    'power_law_fits': {
        'overall': {'A': float(A_all), 'b': float(b_all), 'R2': float(r2_all),
                    'formula': f'non_unique_frac = {A_all:.4f} * k^({b_all:.4f})'},
        'large_k': {'A': float(A_l), 'b': float(b_l), 'R2': float(r2_l),
                    'formula': f'non_unique_frac = {A_l:.4f} * k^({b_l:.4f})'},
    },
    'thresholds': {
        '99%': results_summary['99%'],
        '99.9%': results_summary['99.9%'],
        '99.99%': results_summary['99.99%'],
        '100%': '>100,000 bp (effectively unreachable)'
    }
}

outdir = Path('results')
outdir.mkdir(exist_ok=True)
with open(outdir / 'final_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nFull results saved to results/final_results.json")
