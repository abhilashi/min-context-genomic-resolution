#!/usr/bin/env bash
# Download human reference genome GRCh38 (hg38) from NCBI
# Primary assembly only (no alt contigs), ~900MB compressed, ~3.1GB uncompressed

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
GENOME_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
GENOME_FILE="$DATA_DIR/GRCh38.fa.gz"
GENOME_UNZIPPED="$DATA_DIR/GRCh38.fa"

echo "=== Downloading Human Reference Genome GRCh38 ==="
echo "Destination: $DATA_DIR"

if [ -f "$GENOME_UNZIPPED" ]; then
    echo "Genome already downloaded and unzipped: $GENOME_UNZIPPED"
    exit 0
fi

if [ -f "$GENOME_FILE" ]; then
    echo "Compressed genome found, skipping download."
else
    echo "Downloading from NCBI (~900MB)..."
    curl -L -o "$GENOME_FILE" "$GENOME_URL"
    echo "Download complete."
fi

echo "Decompressing..."
gunzip -k "$GENOME_FILE"
mv "$DATA_DIR/GRCh38.fa.gz" "$DATA_DIR/GRCh38.fa.gz"  # keep compressed copy
echo "Done. Genome at: $GENOME_UNZIPPED"
echo "Size: $(du -h "$GENOME_UNZIPPED" | cut -f1)"
