#!/usr/bin/env python3
"""
K-mer Uniqueness Analysis for the Human Genome

Determines the minimum context size (k-mer length) needed to uniquely
identify each base pair position in the genome.

Algorithm:
  For each k in a range of values, count all k-mers and determine what
  fraction of genome positions are covered by a k-mer that appears exactly
  once (unique). Uses 2-bit encoding and numpy-based sorting for efficiency.

  For the exact per-position minimum unique context, we use an incremental
  approach: start at small k, find positions not yet uniquely identified,
  and increase k until thresholds (99%, 99.9%, 100%) are reached.

Memory-efficient approach:
  - Process in chunks for k-mer extraction
  - Use 2-bit encoding: A=0, C=1, G=2, T=3
  - For k<=32, each k-mer fits in a 64-bit integer
  - Sort k-mer array to find unique k-mers in O(n log n)
"""

import sys
import os
import time
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


# 2-bit encoding: A=0, C=1, G=2, T=3
ENCODE = np.zeros(256, dtype=np.uint8)
ENCODE[ord('A')] = 0; ENCODE[ord('a')] = 0
ENCODE[ord('C')] = 1; ENCODE[ord('c')] = 1
ENCODE[ord('G')] = 2; ENCODE[ord('g')] = 2
ENCODE[ord('T')] = 3; ENCODE[ord('t')] = 3
ENCODE[ord('N')] = 255; ENCODE[ord('n')] = 255  # sentinel for N


def parse_fasta(filepath):
    """Parse a FASTA file, yielding (name, sequence) tuples."""
    name = None
    seq_parts = []

    open_func = open
    if filepath.endswith('.gz'):
        import gzip
        open_func = lambda f: gzip.open(f, 'rt')

    with open_func(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(seq_parts)
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        yield name, ''.join(seq_parts)


def load_genome(filepath, chromosomes=None):
    """
    Load genome sequence from FASTA file.
    Returns concatenated sequence (uppercase) and list of (name, start, end) tuples.
    Chromosomes are separated by 'N' characters to prevent cross-chromosome k-mers.
    """
    print(f"Loading genome from {filepath}...")
    regions = []
    all_seqs = []
    total_len = 0

    for name, seq in parse_fasta(filepath):
        # Filter to main chromosomes if specified
        if chromosomes is not None:
            # Accept chr1, chr2, ..., chrX, chrY, chrM or NC_ accessions
            short_name = name.split('.')[0]
            if short_name not in chromosomes and name not in chromosomes:
                continue

        seq = seq.upper()
        start = total_len
        all_seqs.append(seq)
        total_len += len(seq)
        regions.append((name, start, start + len(seq)))
        print(f"  Loaded {name}: {len(seq):,} bp (total: {total_len:,})")

        # Add separator Ns between chromosomes
        all_seqs.append('N' * 100)
        total_len += 100

    genome = ''.join(all_seqs)
    print(f"Total genome length: {len(genome):,} bp")
    return genome, regions


def encode_sequence(seq):
    """Convert sequence string to numpy array of 2-bit encoded values."""
    seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    return ENCODE[seq_bytes]


def extract_kmers_encoded(encoded_seq, k):
    """
    Extract all k-mers as 64-bit integers from 2-bit encoded sequence.
    Returns array of k-mer values and array of valid positions.
    K-mers containing N (value 255) are excluded.

    For k <= 32 only (fits in uint64).
    """
    n = len(encoded_seq)
    if k > 32:
        raise ValueError("k > 32 not supported with 64-bit encoding. Use hash-based approach.")

    if n < k:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)

    # Find positions with N
    has_n = (encoded_seq == 255)

    # Build k-mer integers using rolling hash
    num_kmers = n - k + 1

    # Initialize: compute first k-mer
    kmers = np.zeros(num_kmers, dtype=np.uint64)
    valid = np.ones(num_kmers, dtype=bool)

    # Mark invalid positions (any k-mer window containing an N)
    # Use cumulative sum of N positions for efficient window checking
    n_cumsum = np.cumsum(has_n.astype(np.int32))
    n_in_window = np.zeros(num_kmers, dtype=np.int32)
    n_in_window = n_cumsum[k-1:] - np.concatenate([[0], n_cumsum[:num_kmers-1]])
    valid = (n_in_window == 0)

    # Compute k-mer integers for valid positions
    # Use vectorized approach: shift and OR
    enc_clean = encoded_seq.astype(np.uint64)
    enc_clean[has_n] = 0  # Replace N with 0 for computation (won't matter, marked invalid)

    # Build k-mers by shifting
    kmers = enc_clean[:num_kmers].copy()
    for i in range(1, k):
        kmers = (kmers << np.uint64(2)) | enc_clean[i:i+num_kmers]

    # Return only valid k-mers with their positions
    valid_positions = np.where(valid)[0]
    valid_kmers = kmers[valid]

    return valid_kmers, valid_positions


def extract_kmers_hash(encoded_seq, k):
    """
    Extract k-mers for k > 32 using hash-based approach.
    Uses a rolling hash (similar to Rabin-Karp) with 128-bit representation.

    For very large k, we use two 64-bit hashes to reduce collision probability.
    """
    n = len(encoded_seq)
    if n < k:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)

    has_n = (encoded_seq == 255)
    num_kmers = n - k + 1

    # Check for N in each window
    n_cumsum = np.cumsum(has_n.astype(np.int32))
    n_in_window = n_cumsum[k-1:] - np.concatenate([[0], n_cumsum[:num_kmers-1]])
    valid = (n_in_window == 0)

    # Use a rolling polynomial hash with two different bases for collision resistance
    # Hash1: base = 4, mod = 2^64 (natural overflow)
    # Hash2: base = 5, mod = 2^64
    BASE1 = np.uint64(4)
    BASE2 = np.uint64(5)

    enc_clean = encoded_seq.astype(np.uint64)
    enc_clean[has_n] = 0

    # Compute power of base for removal of leading digit
    pow1 = np.uint64(1)
    pow2 = np.uint64(1)
    for _ in range(k - 1):
        pow1 = pow1 * BASE1
        pow2 = pow2 * BASE2

    # Compute first hash
    h1 = np.uint64(0)
    h2 = np.uint64(0)
    for i in range(k):
        h1 = h1 * BASE1 + enc_clean[i]
        h2 = h2 * BASE2 + enc_clean[i]

    hash1 = np.zeros(num_kmers, dtype=np.uint64)
    hash2 = np.zeros(num_kmers, dtype=np.uint64)
    hash1[0] = h1
    hash2[0] = h2

    # Rolling hash
    for i in range(1, num_kmers):
        h1 = (h1 - enc_clean[i-1] * pow1) * BASE1 + enc_clean[i+k-1]
        h2 = (h2 - enc_clean[i-1] * pow2) * BASE2 + enc_clean[i+k-1]
        hash1[i] = h1
        hash2[i] = h2

    # Combine into single sortable value (use hash1 as primary, hash2 for collision check)
    valid_positions = np.where(valid)[0]
    valid_h1 = hash1[valid]
    valid_h2 = hash2[valid]

    # Return combined hash as structured array
    combined = np.zeros(len(valid_positions), dtype=[('h1', np.uint64), ('h2', np.uint64)])
    combined['h1'] = valid_h1
    combined['h2'] = valid_h2

    return combined, valid_positions


def count_unique_positions_small_k(encoded_seq, k):
    """
    For k <= 32: extract k-mers as integers, sort, count unique positions.
    Returns (num_unique_positions, num_valid_positions, total_positions).
    """
    kmers, positions = extract_kmers_encoded(encoded_seq, k)

    if len(kmers) == 0:
        return 0, 0, len(encoded_seq) - k + 1

    num_valid = len(kmers)

    # Sort k-mers (keeping track of which are unique)
    sort_idx = np.argsort(kmers)
    sorted_kmers = kmers[sort_idx]

    # Find unique k-mers (appear exactly once)
    n = len(sorted_kmers)
    is_unique = np.ones(n, dtype=bool)

    # Mark duplicates
    same_as_next = np.zeros(n, dtype=bool)
    same_as_next[:-1] = (sorted_kmers[:-1] == sorted_kmers[1:])
    same_as_prev = np.zeros(n, dtype=bool)
    same_as_prev[1:] = same_as_next[:-1]

    is_unique = ~(same_as_next | same_as_prev)

    num_unique = np.sum(is_unique)
    total_positions = len(encoded_seq) - k + 1

    return int(num_unique), int(num_valid), int(total_positions)


def count_unique_positions_large_k(encoded_seq, k):
    """
    For k > 32: use hash-based approach.
    Returns (num_unique_positions, num_valid_positions, total_positions).
    """
    combined, positions = extract_kmers_hash(encoded_seq, k)

    if len(combined) == 0:
        return 0, 0, len(encoded_seq) - k + 1

    num_valid = len(combined)

    # Sort by h1, then h2
    sort_idx = np.argsort(combined, order=('h1', 'h2'))
    sorted_h = combined[sort_idx]

    n = len(sorted_h)
    is_unique = np.ones(n, dtype=bool)

    same_as_next = np.zeros(n, dtype=bool)
    same_as_next[:-1] = ((sorted_h['h1'][:-1] == sorted_h['h1'][1:]) &
                          (sorted_h['h2'][:-1] == sorted_h['h2'][1:]))
    same_as_prev = np.zeros(n, dtype=bool)
    same_as_prev[1:] = same_as_next[:-1]

    is_unique = ~(same_as_next | same_as_prev)

    num_unique = np.sum(is_unique)
    total_positions = len(encoded_seq) - k + 1

    return int(num_unique), int(num_valid), int(total_positions)


def analyze_kmer_uniqueness(genome_seq, k_values, output_dir=None):
    """
    Main analysis: for each k, compute the fraction of positions
    with unique k-mers.

    Returns dict mapping k -> {unique, valid, total, pct_unique, pct_of_valid}
    """
    print(f"\nEncoding genome sequence ({len(genome_seq):,} bp)...")
    encoded = encode_sequence(genome_seq)

    # Count non-N bases
    non_n = np.sum(encoded != 255)
    print(f"Non-N bases: {non_n:,} ({100*non_n/len(encoded):.1f}%)")

    results = {}

    print(f"\n{'k':>6} | {'Unique':>14} | {'Valid':>14} | {'Total':>14} | {'% Unique (valid)':>18} | {'% Unique (total)':>18} | {'Time':>8}")
    print("-" * 105)

    for k in sorted(k_values):
        t0 = time.time()

        if k <= 32:
            unique, valid, total = count_unique_positions_small_k(encoded, k)
        else:
            unique, valid, total = count_unique_positions_large_k(encoded, k)

        elapsed = time.time() - t0
        pct_valid = 100.0 * unique / valid if valid > 0 else 0
        pct_total = 100.0 * unique / total if total > 0 else 0

        results[k] = {
            'k': k,
            'unique_positions': unique,
            'valid_positions': valid,
            'total_positions': total,
            'pct_unique_of_valid': round(pct_valid, 4),
            'pct_unique_of_total': round(pct_total, 4),
            'elapsed_seconds': round(elapsed, 2)
        }

        print(f"{k:>6} | {unique:>14,} | {valid:>14,} | {total:>14,} | {pct_valid:>17.4f}% | {pct_total:>17.4f}% | {elapsed:>7.1f}s")

    # Find threshold k values
    print("\n=== Threshold Analysis ===")
    thresholds = [99.0, 99.9, 100.0]
    threshold_results = {}

    sorted_results = sorted(results.items())

    for thresh in thresholds:
        found_k = None
        for k, r in sorted_results:
            if r['pct_unique_of_valid'] >= thresh:
                found_k = k
                break

        if found_k is not None:
            print(f"  {thresh:>5.1f}% unique: k >= {found_k}")
            threshold_results[str(thresh)] = found_k
        else:
            max_k = max(k_values)
            best_pct = results[max_k]['pct_unique_of_valid']
            print(f"  {thresh:>5.1f}% unique: NOT reached at k={max_k} (best: {best_pct:.4f}%)")
            threshold_results[str(thresh)] = f">{max_k}"

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            'genome_length': len(genome_seq),
            'non_n_bases': int(non_n),
            'k_values': results,
            'thresholds': threshold_results
        }

        outfile = output_dir / 'kmer_uniqueness_results.json'
        with open(outfile, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {outfile}")

    return results, threshold_results


def analyze_per_position_min_k(genome_seq, max_k=32, sample_size=None):
    """
    For each position (or a sample), find the MINIMUM k such that the
    k-mer starting at that position is unique in the genome.

    This gives the exact per-position minimum context requirement.
    Uses incremental approach: start at k=11, increase until position is resolved.

    Returns histogram of minimum k values.
    """
    print(f"\n=== Per-Position Minimum Unique Context Analysis ===")
    encoded = encode_sequence(genome_seq)
    n = len(encoded)

    # Track which positions still need resolution
    if sample_size and sample_size < n:
        # Sample random positions (avoiding N regions)
        non_n_positions = np.where(encoded != 255)[0]
        if sample_size > len(non_n_positions):
            sample_size = len(non_n_positions)
        rng = np.random.RandomState(42)
        sample_positions = np.sort(rng.choice(non_n_positions, sample_size, replace=False))
        print(f"Sampling {sample_size:,} positions out of {len(non_n_positions):,} non-N positions")
    else:
        sample_positions = None  # Use all positions
        sample_size = n

    # min_k[i] = minimum k for position i to be unique (0 = not yet resolved)
    min_k = np.zeros(sample_size, dtype=np.int32)
    unresolved = np.ones(sample_size, dtype=bool)

    for k in range(11, max_k + 1):
        if not np.any(unresolved):
            break

        t0 = time.time()

        # Get all k-mers
        if k <= 32:
            kmers, positions = extract_kmers_encoded(encoded, k)
        else:
            kmers_hash, positions = extract_kmers_hash(encoded, k)

        if len(positions) == 0:
            continue

        # Find unique k-mers
        if k <= 32:
            sort_idx = np.argsort(kmers)
            sorted_kmers = kmers[sort_idx]
            sorted_positions = positions[sort_idx]

            nm = len(sorted_kmers)
            same_next = np.zeros(nm, dtype=bool)
            same_next[:-1] = (sorted_kmers[:-1] == sorted_kmers[1:])
            same_prev = np.zeros(nm, dtype=bool)
            same_prev[1:] = same_next[:-1]

            unique_mask = ~(same_next | same_prev)
            unique_positions_set = set(sorted_positions[unique_mask])
        else:
            sort_idx = np.argsort(kmers_hash, order=('h1', 'h2'))
            sorted_h = kmers_hash[sort_idx]
            sorted_positions = positions[sort_idx]

            nm = len(sorted_h)
            same_next = np.zeros(nm, dtype=bool)
            same_next[:-1] = ((sorted_h['h1'][:-1] == sorted_h['h1'][1:]) &
                               (sorted_h['h2'][:-1] == sorted_h['h2'][1:]))
            same_prev = np.zeros(nm, dtype=bool)
            same_prev[1:] = same_next[:-1]

            unique_mask = ~(same_next | same_prev)
            unique_positions_set = set(sorted_positions[unique_mask])

        # Check which unresolved positions are now unique
        newly_resolved = 0
        if sample_positions is not None:
            for i in range(sample_size):
                if unresolved[i] and sample_positions[i] in unique_positions_set:
                    min_k[i] = k
                    unresolved[i] = False
                    newly_resolved += 1
        else:
            for i in range(sample_size):
                if unresolved[i] and i in unique_positions_set:
                    min_k[i] = k
                    unresolved[i] = False
                    newly_resolved += 1

        elapsed = time.time() - t0
        n_resolved = np.sum(~unresolved)
        pct = 100.0 * n_resolved / sample_size
        print(f"  k={k:>3}: {newly_resolved:>10,} newly resolved, {n_resolved:>10,} total ({pct:.4f}%) [{elapsed:.1f}s]")

    # Histogram of minimum k values
    still_unresolved = np.sum(unresolved)
    print(f"\n  Positions still unresolved at k={max_k}: {still_unresolved:,} ({100*still_unresolved/sample_size:.4f}%)")

    # Compute percentiles
    resolved_k = min_k[~unresolved]
    if len(resolved_k) > 0:
        print(f"\n  Distribution of minimum unique context (resolved positions):")
        for pct in [50, 75, 90, 95, 99, 99.9]:
            val = np.percentile(resolved_k, pct)
            print(f"    {pct:>5.1f}th percentile: k = {val:.0f}")
        print(f"    Mean: k = {np.mean(resolved_k):.1f}")
        print(f"    Max:  k = {np.max(resolved_k)}")

    return min_k, unresolved


def main():
    parser = argparse.ArgumentParser(
        description='Analyze k-mer uniqueness in a genome sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with a single chromosome
  python kmer_uniqueness.py ../data/GRCh38.fa --chromosomes chr22 --k-range 11 32

  # Full genome analysis at specific k values
  python kmer_uniqueness.py ../data/GRCh38.fa --k-values 11 15 20 25 30

  # Per-position analysis with sampling
  python kmer_uniqueness.py ../data/GRCh38.fa --chromosomes chr22 --per-position --sample 100000
        """
    )
    parser.add_argument('genome', help='Path to genome FASTA file (.fa or .fa.gz)')
    parser.add_argument('--chromosomes', nargs='+', default=None,
                        help='Specific chromosomes to analyze (default: all)')
    parser.add_argument('--k-values', nargs='+', type=int, default=None,
                        help='Specific k values to test')
    parser.add_argument('--k-range', nargs=2, type=int, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Range of k values to test')
    parser.add_argument('--per-position', action='store_true',
                        help='Compute per-position minimum unique context')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size for per-position analysis')
    parser.add_argument('--output', default='../results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Determine k values to test
    if args.k_values:
        k_values = args.k_values
    elif args.k_range:
        k_values = list(range(args.k_range[0], args.k_range[1] + 1))
    else:
        # Default: key k values for threshold detection
        k_values = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32]

    # Load genome
    genome_seq, regions = load_genome(args.genome, args.chromosomes)

    # Run k-mer uniqueness analysis
    results, thresholds = analyze_kmer_uniqueness(genome_seq, k_values, args.output)

    # Optionally run per-position analysis
    if args.per_position:
        max_k = max(k_values)
        min_k_arr, unresolved = analyze_per_position_min_k(
            genome_seq, max_k=max_k, sample_size=args.sample
        )

    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()
