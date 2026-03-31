#!/usr/bin/env python3
"""
Full Genome K-mer Uniqueness Analysis using Jellyfish

Determines the minimum context size (k-mer length) needed to uniquely
identify each base pair position in the human genome.

For each k: uses jellyfish to count k-mers, then extracts uniqueness stats.
Uses appropriate hash sizes to avoid memory bloat.
"""

import subprocess
import sys
import os
import json
import time
import tempfile
import shutil
from pathlib import Path


def get_hash_size(k, genome_size=3_100_000_000):
    """Choose appropriate jellyfish hash size based on k."""
    # Number of possible k-mers = min(4^k, genome_size)
    possible = min(4**k, genome_size)
    # Use ~2x the expected distinct k-mers, capped at 2B entries
    size = min(int(possible * 1.5), 2_000_000_000)
    # Floor at 10M
    return str(max(size, 10_000_000))


def run_jellyfish(fasta_path, k, tmpdir, threads=8):
    """Run jellyfish count + stats for a given k. Returns stats dict or None."""
    db_path = os.path.join(tmpdir, f"mer_{k}.jf")
    hash_size = get_hash_size(k)

    # Count
    cmd = ["jellyfish", "count", "-m", str(k), "-s", hash_size,
           "-t", str(threads), "-o", db_path, fasta_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  jellyfish count error (k={k}): {r.stderr}", file=sys.stderr)
        return None

    # Histo (just need count=1 and total)
    cmd = ["jellyfish", "histo", "-l", "1", "-h", "100000", db_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  jellyfish histo error (k={k}): {r.stderr}", file=sys.stderr)
        return None

    histo = {}
    for line in r.stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) == 2:
            histo[int(parts[0])] = int(parts[1])

    unique_kmers = histo.get(1, 0)
    distinct_kmers = sum(histo.values())
    total_kmers = sum(c * n for c, n in histo.items())

    # Clean up db
    for f in Path(tmpdir).glob(f"mer_{k}*"):
        f.unlink()

    return {
        'unique_kmers': unique_kmers,
        'distinct_kmers': distinct_kmers,
        'total_kmers': total_kmers,
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
    parser = argparse.ArgumentParser(description='K-mer uniqueness analysis')
    parser.add_argument('genome', help='Path to genome FASTA file')
    parser.add_argument('--k-values', nargs='+', type=int, default=None)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    if args.k_values:
        k_values = sorted(args.k_values)
    else:
        k_values = [11, 13, 15, 17, 20, 24, 28, 32, 40, 50,
                    75, 100, 150, 200, 300, 500, 750, 1000]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Counting bases in {args.genome}...")
    total_bases, non_n_bases = get_genome_stats(args.genome)
    print(f"  Total bases: {total_bases:,}")
    print(f"  Non-N bases: {non_n_bases:,} ({100*non_n_bases/total_bases:.1f}%)")

    tmpdir = tempfile.mkdtemp(prefix="jf_")
    results = {}

    print(f"\n{'k':>6} | {'Unique k-mers':>15} | {'Distinct':>15} | {'Total pos':>15} | {'% Unique':>12} | {'Hash size':>12} | {'Time':>8}")
    print("-" * 105)

    for k in k_values:
        t0 = time.time()
        hs = get_hash_size(k)

        stats = run_jellyfish(args.genome, k, tmpdir, args.threads)
        if stats is None:
            continue

        elapsed = time.time() - t0
        pct = 100.0 * stats['unique_kmers'] / stats['total_kmers'] if stats['total_kmers'] > 0 else 0

        results[k] = {
            'k': k,
            'unique_kmers': stats['unique_kmers'],
            'distinct_kmers': stats['distinct_kmers'],
            'total_positions': stats['total_kmers'],
            'pct_unique': round(pct, 6),
            'elapsed_seconds': round(elapsed, 1)
        }

        print(f"{k:>6} | {stats['unique_kmers']:>15,} | {stats['distinct_kmers']:>15,} | {stats['total_kmers']:>15,} | {pct:>11.4f}% | {hs:>12} | {elapsed:>7.1f}s")
        sys.stdout.flush()

    # Threshold analysis
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)
    thresholds = [50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 99.99, 100.0]
    threshold_results = {}
    sorted_r = sorted(results.items())

    for thresh in thresholds:
        found_k = None
        for k, r in sorted_r:
            if r['pct_unique'] >= thresh:
                found_k = k
                break
        if found_k:
            print(f"  {thresh:>6.2f}% unique: k >= {found_k}")
            threshold_results[str(thresh)] = found_k
        else:
            max_k = max(k_values)
            best = results.get(max_k, {}).get('pct_unique', 0)
            print(f"  {thresh:>6.2f}% unique: NOT reached at k={max_k} (best: {best:.4f}%)")
            threshold_results[str(thresh)] = f">{max_k}"

    # Power law extrapolation
    print("\n" + "=" * 60)
    print("POWER LAW EXTRAPOLATION")
    print("=" * 60)
    try:
        import numpy as np
        ks_fit = []
        fracs_fit = []
        for k, r in sorted_r:
            if 20 <= k <= 1000 and 0 < r['pct_unique'] < 100:
                ks_fit.append(k)
                fracs_fit.append(1.0 - r['pct_unique'] / 100.0)

        if len(ks_fit) >= 3:
            log_k = np.log(ks_fit)
            log_f = np.log(fracs_fit)
            b, a = np.polyfit(log_k, log_f, 1)
            A = np.exp(a)
            print(f"  Fit: non_unique_fraction = {A:.6f} * k^({b:.4f})")

            for target in [99.0, 99.9, 99.99, 100.0]:
                frac = 1.0 - target / 100.0
                if frac > 0 and A > 0:
                    k_est = (frac / A) ** (1.0 / b)
                    print(f"  Estimated k for {target:>6.2f}%: ~{k_est:,.0f} bp")
    except Exception as e:
        print(f"  Extrapolation failed: {e}")

    # Save
    output = {
        'genome_file': str(args.genome),
        'total_bases': total_bases,
        'non_n_bases': non_n_bases,
        'k_results': {str(k): v for k, v in results.items()},
        'thresholds': threshold_results
    }
    outfile = output_dir / 'full_genome_kmer_uniqueness.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
