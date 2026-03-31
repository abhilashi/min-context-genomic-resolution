#!/usr/bin/env python3
"""
K-mer Uniqueness Analysis using KMC (disk-based k-mer counter)

KMC is optimized for the human genome scale and uses disk for intermediate
storage, avoiding RAM limitations. It handles k up to 256 natively.

For each k:
  1. kmc -k{k} -ci1 -cs65535 genome.fa output tmpdir
  2. kmc_tools transform output histogram histo.txt
  3. Parse histogram: count=1 → unique k-mers

For k > 256, we use jellyfish or a sampling approach.
"""

import subprocess
import sys
import os
import json
import time
import tempfile
import shutil
from pathlib import Path


def run_kmc(fasta_path, k, tmpdir, threads=8):
    """
    Run KMC k-mer counting for given k.
    Returns dict with unique_kmers, distinct_kmers, total_kmers.
    """
    db_prefix = os.path.join(tmpdir, f"kmc_k{k}")
    histo_file = os.path.join(tmpdir, f"histo_k{k}.txt")
    kmc_tmp = os.path.join(tmpdir, f"kmc_tmp_k{k}")
    os.makedirs(kmc_tmp, exist_ok=True)

    # KMC count
    # -k: k-mer length
    # -ci1: min count inclusion = 1 (include all)
    # -cs65535: max count storage = 65535
    # -t: threads
    # -fm: input is multi-FASTA
    cmd = [
        "kmc", f"-k{k}", "-ci1", "-cs65535",
        f"-t{threads}", "-fm", "-m12",
        fasta_path, db_prefix, kmc_tmp
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  KMC error (k={k}): {r.stderr}", file=sys.stderr)
        return None

    # Parse KMC stdout for stats
    total_kmers = 0
    unique_kmers_kmc = 0
    distinct_kmers = 0
    for line in r.stdout.split('\n'):
        line = line.strip()
        if 'Total no. of k-mers' in line:
            total_kmers = int(line.split(':')[-1].strip())
        elif 'No. of unique k-mers' in line:
            distinct_kmers = int(line.split(':')[-1].strip())

    # Generate histogram
    cmd_histo = [
        "kmc_tools", "transform",
        db_prefix, "histogram", histo_file
    ]
    r2 = subprocess.run(cmd_histo, capture_output=True, text=True)
    if r2.returncode != 0:
        print(f"  kmc_tools error (k={k}): {r2.stderr}", file=sys.stderr)
        # Try to parse from KMC output instead
        return None

    # Parse histogram
    histo = {}
    with open(histo_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                count, num = int(parts[0]), int(parts[1])
                histo[count] = num

    unique_kmers = histo.get(1, 0)  # k-mers appearing exactly once

    # Recompute totals from histogram for accuracy
    distinct_from_histo = sum(histo.values())
    total_from_histo = sum(c * n for c, n in histo.items())

    # Cleanup KMC files
    for pattern in [f"kmc_k{k}*", f"histo_k{k}*"]:
        for f in Path(tmpdir).glob(pattern):
            f.unlink(missing_ok=True)
    shutil.rmtree(kmc_tmp, ignore_errors=True)

    return {
        'unique_kmers': unique_kmers,
        'distinct_kmers': distinct_from_histo,
        'total_kmers': total_from_histo,
        'histogram_top10': {str(k): v for k, v in sorted(histo.items())[:10]}
    }


def run_jellyfish_large_k(fasta_path, k, tmpdir, threads=8):
    """Fallback to jellyfish for k > 256 (KMC max)."""
    db_path = os.path.join(tmpdir, f"jf_k{k}.jf")

    # For large k, most k-mers are unique, so hash table can be moderate
    hash_size = "500000000"

    cmd = ["jellyfish", "count", "-m", str(k), "-s", hash_size,
           "-t", str(threads), "-o", db_path, fasta_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  jellyfish error (k={k}): {r.stderr}", file=sys.stderr)
        return None

    cmd = ["jellyfish", "histo", "-l", "1", "-h", "100000", db_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None

    histo = {}
    for line in r.stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) == 2:
            histo[int(parts[0])] = int(parts[1])

    for f in Path(tmpdir).glob(f"jf_k{k}*"):
        f.unlink(missing_ok=True)

    return {
        'unique_kmers': histo.get(1, 0),
        'distinct_kmers': sum(histo.values()),
        'total_kmers': sum(c * n for c, n in histo.items()),
        'histogram_top10': {str(k): v for k, v in sorted(histo.items())[:10]}
    }


def get_genome_stats(fasta_path):
    """Count total and non-N bases."""
    total = 0
    non_n = 0
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith('>'):
                seq = line.strip().upper()
                total += len(seq)
                non_n += sum(1 for c in seq if c in 'ACGT')
    return total, non_n


def main():
    import argparse
    parser = argparse.ArgumentParser(description='K-mer uniqueness using KMC')
    parser.add_argument('genome', help='Path to genome FASTA file')
    parser.add_argument('--k-values', nargs='+', type=int, default=None)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    if args.k_values:
        k_values = sorted(args.k_values)
    else:
        k_values = [11, 13, 15, 17, 20, 24, 28, 32, 40, 50,
                    75, 100, 150, 200, 250, 300, 400, 500,
                    750, 1000]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Counting bases in {args.genome}...")
    total_bases, non_n_bases = get_genome_stats(args.genome)
    print(f"  Total bases: {total_bases:,}")
    print(f"  Non-N bases: {non_n_bases:,} ({100*non_n_bases/total_bases:.1f}%)")
    print(f"  N bases:     {total_bases - non_n_bases:,}")

    tmpdir = tempfile.mkdtemp(prefix="kmc_work_")
    results = {}

    print(f"\n{'k':>6} | {'Unique (cnt=1)':>15} | {'Distinct':>15} | {'Total k-mers':>15} | {'% Unique':>12} | {'Time':>8}")
    print("-" * 95)

    for k in k_values:
        t0 = time.time()

        if k <= 256:
            stats = run_kmc(args.genome, k, tmpdir, args.threads)
        else:
            stats = run_jellyfish_large_k(args.genome, k, tmpdir, args.threads)

        if stats is None:
            print(f"{k:>6} | {'ERROR':>15} |")
            continue

        elapsed = time.time() - t0
        pct = 100.0 * stats['unique_kmers'] / stats['total_kmers'] if stats['total_kmers'] > 0 else 0

        results[k] = {
            'k': k,
            'unique_kmers': stats['unique_kmers'],
            'distinct_kmers': stats['distinct_kmers'],
            'total_positions': stats['total_kmers'],
            'pct_unique': round(pct, 6),
            'elapsed_seconds': round(elapsed, 1),
        }

        print(f"{k:>6} | {stats['unique_kmers']:>15,} | {stats['distinct_kmers']:>15,} | {stats['total_kmers']:>15,} | {pct:>11.4f}% | {elapsed:>7.1f}s")
        sys.stdout.flush()

    # Threshold analysis
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS: Minimum k for X% of positions unique")
    print("=" * 60)
    thresholds = [50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 99.99, 100.0]
    threshold_results = {}
    sorted_r = sorted(results.items())

    for thresh in thresholds:
        found_k = None
        for kv, r in sorted_r:
            if r['pct_unique'] >= thresh:
                found_k = kv
                break
        if found_k:
            print(f"  {thresh:>6.2f}% unique: k >= {found_k}")
            threshold_results[str(thresh)] = found_k
        else:
            max_k = max(k_values)
            best = results.get(max_k, {}).get('pct_unique', 0)
            print(f"  {thresh:>6.2f}% unique: NOT reached at k={max_k} (best: {best:.4f}%)")
            threshold_results[str(thresh)] = f">{max_k}"

    # Interpolation between measured k values
    print("\n" + "=" * 60)
    print("INTERPOLATION (between measured points)")
    print("=" * 60)
    try:
        import numpy as np

        # Use points where pct_unique is between 1% and 99.5% for fitting
        ks_fit = []
        fracs_fit = []
        for kv, r in sorted_r:
            if 0.5 < r['pct_unique'] < 99.95:
                ks_fit.append(kv)
                fracs_fit.append(1.0 - r['pct_unique'] / 100.0)

        if len(ks_fit) >= 3:
            log_k = np.log(np.array(ks_fit, dtype=float))
            log_f = np.log(np.array(fracs_fit, dtype=float))

            # Fit piecewise: small k and large k may have different exponents
            # Single power law fit
            b, a = np.polyfit(log_k, log_f, 1)
            A = np.exp(a)

            # R² calculation
            y_pred = a + b * log_k
            ss_res = np.sum((log_f - y_pred) ** 2)
            ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
            r_squared = 1 - ss_res / ss_tot

            print(f"  Power law: non_unique_frac = {A:.6f} * k^({b:.4f})")
            print(f"  R² = {r_squared:.4f}")
            print()

            for target in [99.0, 99.9, 99.99, 100.0]:
                frac = 1.0 - target / 100.0
                if frac > 0 and A > 0:
                    k_est = (frac / A) ** (1.0 / b)
                    print(f"  Estimated k for {target:>6.2f}%: ~{k_est:,.0f} bp")

            # Also fit just the large-k regime (k >= 50)
            ks_large = [kv for kv in ks_fit if kv >= 50]
            fracs_large = [f for kv, f in zip(ks_fit, fracs_fit) if kv >= 50]

            if len(ks_large) >= 2:
                log_kl = np.log(np.array(ks_large, dtype=float))
                log_fl = np.log(np.array(fracs_large, dtype=float))
                bl, al = np.polyfit(log_kl, log_fl, 1)
                Al = np.exp(al)

                print(f"\n  Large-k power law (k>=50): non_unique_frac = {Al:.6f} * k^({bl:.4f})")
                for target in [99.0, 99.9, 99.99]:
                    frac = 1.0 - target / 100.0
                    if frac > 0 and Al > 0:
                        k_est = (frac / Al) ** (1.0 / bl)
                        print(f"  Estimated k for {target:>6.2f}%: ~{k_est:,.0f} bp")

    except Exception as e:
        print(f"  Interpolation error: {e}")

    # Save results
    output = {
        'genome_file': str(args.genome),
        'total_bases': total_bases,
        'non_n_bases': non_n_bases,
        'k_results': {str(k): v for k, v in results.items()},
        'thresholds': threshold_results,
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    outfile = output_dir / 'kmer_uniqueness_hg38.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
