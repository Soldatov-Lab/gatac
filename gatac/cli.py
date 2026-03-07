"""
GATAC Command Line Interface.

Usage:
    gatac convert <input.tsv.gz> [output.parquet]
    gatac tile <input.parquet> [-o output] [-t tile_size] [-m min_frags]
    gatac gene <input.parquet> -g <annotations.gtf> [-o output]
    gatac features <input.h5ad> [-n n_features] [-o output]
    gatac combine <input.h5ad> [input2.h5ad ...] -o <output.h5ad>
    gatac metrics <input.parquet> -g <annotations.gtf> [-o output]
    gatac filter <input.parquet> [--metrics metrics.csv] [--filter "query"]
"""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s' if not verbose else '%(levelname)s [%(name)s]: %(message)s'
    )


def convert_command(args):
    """Handle 'gatac convert' subcommand."""
    import glob
    from .pp.convert import make_parquet, make_parquet_batch

    # Expand inputs - support glob patterns
    input_paths = []
    for inp in args.input:
        if '*' in inp or '?' in inp:
            expanded = sorted(glob.glob(inp))
            if not expanded:
                logging.warning(f"No files matched pattern: {inp}")
            input_paths.extend(expanded)
        else:
            input_paths.append(inp)

    input_paths = [Path(p) for p in input_paths]
    for p in input_paths:
        if not p.exists():
            logging.error(f"Input file not found: {p}")
            sys.exit(1)

    if len(input_paths) == 0:
        logging.error("No input files found")
        sys.exit(1)

    if len(input_paths) == 1:
        if args.output_dir:
            output_path = Path(args.output_dir) / input_paths[0].with_suffix('').with_suffix('.parquet').name
        else:
            output_path = args.output if args.output else None
        make_parquet(
            input_paths[0],
            output_path,
            barcode_prefix=args.barcode_prefix,
        )
    else:
        if args.output:
            logging.error("Use --output-dir instead of --output when converting multiple files")
            sys.exit(1)
        make_parquet_batch(
            input_paths,
            output_dir=args.output_dir,
            workers=args.workers,
            barcode_prefix=args.barcode_prefix,
        )


def tile_command(args):
    """Handle 'gatac tile' subcommand."""
    from .pp.tile import make_tile_matrix

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Resolve genome if it's a file
    genome_arg = args.genome
    if Path(genome_arg).exists():
        chrom_sizes = {}
        try:
            with open(genome_arg, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        chrom_sizes[parts[0]] = int(parts[1])
            genome_arg = chrom_sizes
        except Exception as e:
            logging.error(f"Error reading chromosome sizes file: {e}")
            sys.exit(1)

    try:
        make_tile_matrix(
            input_parquet=input_path,
            chrom_sizes=genome_arg,
            output_path=args.output,
            tile_size=args.tile_size,
            min_fragments_per_cell=args.min_fragments,
            exclude_chroms=args.exclude_chroms,
            metrics=args.metrics,
            filter_query=args.filter_query,
            count_strategy=args.count_strategy,
            barcode_prefix=args.barcode_prefix,
            low_memory=args.low_memory,
        )
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error creating tile matrix: {e}")
        sys.exit(1)


def features_command(args):
    """Handle 'gatac features' subcommand."""
    import glob
    import scanpy as sc
    from .pp.features import select_features, select_features_multi

    # Expand inputs - support glob patterns
    input_paths = []
    for inp in args.input:
        if '*' in inp or '?' in inp:
            # Glob pattern
            expanded = sorted(glob.glob(inp))
            if not expanded:
                logging.warning(f"No files matched pattern: {inp}")
            input_paths.extend(expanded)
        else:
            input_paths.append(inp)

    # Validate inputs exist
    input_paths = [Path(p) for p in input_paths]
    for p in input_paths:
        if not p.exists():
            logging.error(f"Input file not found: {p}")
            sys.exit(1)

    if len(input_paths) == 0:
        logging.error("No input files found")
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if len(input_paths) > 1:
        if output_path is None:
            logging.error("--output is required when processing multiple files")
            sys.exit(1)
        # Use multi-file function
        select_features_multi(
            input_paths,
            output_path=output_path,
            n_features=args.n_features,
            binarize=not args.no_binarize,
        )
    else:
        # Single file - use regular function
        if output_path is None:
            output_path = input_paths[0].with_name(input_paths[0].stem + '_selected.h5ad')
        adata = sc.read_h5ad(str(input_paths[0]))
        select_features(
            adata,
            n_features=args.n_features,
            output_path=output_path,
        )


def combine_command(args):
    """Handle 'gatac combine' subcommand."""
    import glob
    from .pp.process import combine

    # Expand inputs - support glob patterns
    input_paths = []
    for inp in args.input:
        if '*' in inp or '?' in inp:
            # Glob pattern
            expanded = sorted(glob.glob(inp))
            if not expanded:
                logging.warning(f"No files matched pattern: {inp}")
            input_paths.extend(expanded)
        else:
            input_paths.append(inp)

    # Validate inputs exist
    input_paths = [Path(p) for p in input_paths]
    for p in input_paths:
        if not p.exists():
            logging.error(f"Input file not found: {p}")
            sys.exit(1)

    if len(input_paths) == 0:
        logging.error("No input files found")
        sys.exit(1)

    if args.output is None:
        logging.error("--output is required for combine command")
        sys.exit(1)

    combine(
        input_paths,
        output_path=args.output,
    )


def metrics_command(args):
    """Handle 'gatac metrics' subcommand."""
    input_path = Path(args.input)
    gtf_path = Path(args.gtf)
    
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)
    if not gtf_path.exists():
        logging.error(f"GTF file not found: {gtf_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix('').with_name(input_path.stem + '_metrics.csv')

    # Use Polars GPU streaming for out-of-core processing
    from .pp.metrics import load_tss_from_gtf, compute_metrics
    
    logging.info(f"Loading fragments from {input_path}")
    tss_df = load_tss_from_gtf(gtf_path)
    results = compute_metrics(
        input_path,
        tss_df,
        min_unique_frags=args.min_frags,
        row_groups_per_batch=args.batch_size,
    )
    
    logging.info(f"Saving results to {output_path}")
    results.to_csv(str(output_path), index=False)
    logging.info(f"Successfully processed {len(results):,} cells.")


def gene_command(args):
    """Handle 'gatac gene' subcommand."""
    from .pp.gene import make_gene_matrix

    input_path = Path(args.input)
    gtf_path = Path(args.gtf)
    
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)
    if not gtf_path.exists():
        logging.error(f"GTF file not found: {gtf_path}")
        sys.exit(1)

    try:
        make_gene_matrix(
            input_parquet=input_path,
            gene_anno=gtf_path,
            output_path=args.output,
            id_type=args.id_type,
            upstream=args.upstream,
            downstream=args.downstream,
            include_gene_body=args.include_gene_body,
            min_fragments_per_cell=args.min_fragments,
            exclude_chroms=args.exclude_chroms,
            metrics=args.metrics,
            filter_query=args.filter_query,
            barcode_prefix=args.barcode_prefix,
            low_memory=args.low_memory,
        )
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error creating gene matrix: {e}")
        sys.exit(1)


def filter_command(args):
    """Handle 'gatac filter' subcommand."""
    import glob
    
    # Expand inputs - support glob patterns
    input_paths = []
    for inp in args.input:
        if '*' in inp or '?' in inp:
            expanded = sorted(glob.glob(inp))
            if not expanded:
                logging.warning(f"No files matched pattern: {inp}")
            input_paths.extend(expanded)
        else:
            input_paths.append(inp)

    # Validate inputs exist
    input_paths = [Path(p) for p in input_paths]
    for p in input_paths:
        if not p.exists():
            logging.error(f"Input file not found: {p}")
            sys.exit(1)

    if len(input_paths) == 0:
        logging.error("No input files found")
        sys.exit(1)

    # Handle metrics file
    metrics_path = None
    if args.metrics:
        metrics_path = Path(args.metrics)
        if not metrics_path.exists():
            logging.error(f"Metrics file not found: {metrics_path}")
            sys.exit(1)

    # Determine output paths
    if args.output:
        if len(input_paths) > 1:
            logging.error("Cannot specify single output for multiple input files")
            sys.exit(1)
        output_paths = Path(args.output)
    else:
        output_paths = None

    from .pp.filter import filter_fragments

    try:
        filter_fragments(
            input_parquet=input_paths if len(input_paths) > 1 else input_paths[0],
            output_parquet=output_paths,
            metrics=metrics_path,
            min_fragments_per_cell=args.min_fragments,
            filter_query=args.filter_query,
            barcode_prefix=args.barcode_prefix,
            row_groups_per_batch=args.batch_size,
            chrom_sizes=args.genome if hasattr(args, 'genome') and args.genome else None,
        )
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error filtering fragments: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='gatac',
        description='GPU-accelerated ATAC-seq processing toolkit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (debug) output'
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Convert subcommand
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert ATAC fragments TSV.GZ to Parquet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    convert_parser.add_argument(
        'input',
        nargs='+',
        help='Input .tsv.gz file(s) or glob pattern (e.g. "samples/*.tsv.gz")'
    )
    convert_parser.add_argument(
        '-o', '--output',
        help='Output .parquet file (single-file mode only)'
    )
    convert_parser.add_argument(
        '--output-dir',
        help='Output directory for Parquet files (multi-file mode)'
    )
    convert_parser.add_argument(
        '-j', '--workers',
        type=int,
        default=None,
        help='Number of parallel worker processes (default: number of input files, capped at CPU count)'
    )
    convert_parser.add_argument(
        '--barcode-prefix',
        help='Prefix to add to barcodes'
    )
    convert_parser.set_defaults(func=convert_command)

    # Tile subcommand
    tile_parser = subparsers.add_parser(
        'tile',
        help='Process fragments to tile matrix',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    tile_parser.add_argument(
        'input',
        help='Input .parquet file'
    )
    tile_parser.add_argument(
        '-o', '--output',
        help='Output .h5ad file'
    )
    tile_parser.add_argument(
        '-t', '--tile-size',
        type=int,
        default=5000,
        help='Tile size in bp'
    )
    tile_parser.add_argument(
        '-m', '--min-fragments',
        type=int,
        default=100,
        help='Min fragments per cell'
    )
    tile_parser.add_argument(
        '-g', '--genome',
        required=True,
        help='Genome name (e.g., hg38, mm10) or path to chromosome sizes file'
    )
    tile_parser.add_argument(
        '-e', '--exclude-chroms',
        nargs='+',
        default=["chrM", "chrY", "M", "Y"],
        help='Chromosomes to exclude'
    )
    tile_parser.add_argument(
        '--metrics',
        help='Path to CSV file with quality metrics for filtering'
    )
    tile_parser.add_argument(
        '--filter',
        dest='filter_query',
        help='Filtering query string (e.g., "tsse_score > 5")'
    )
    tile_parser.add_argument(
        '--barcode-prefix',
        help='Prefix to add to barcodes'
    )
    tile_parser.add_argument(
        '--low-memory',
        action='store_true',
        help='Use low memory mode for Parquet reading'
    )
    tile_parser.add_argument(
        '--count-strategy',
        choices=['unique', 'count', 'binarize'],
        default='unique',
        help='Counting strategy: unique (SnapATAC2-compatible), count (includes PCR duplicates), or binarize (binary 0/1)'
    )
    tile_parser.set_defaults(func=tile_command)

    # Features subcommand
    features_parser = subparsers.add_parser(
        'features',
        help='GPU-accelerated feature selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    features_parser.add_argument(
        'input',
        nargs='+',
        help='Input .h5ad file(s) or glob pattern (e.g., "path/*.h5ad")'
    )
    features_parser.add_argument(
        '-n', '--n-features',
        type=int,
        default=500000,
        help='Number of features to select'
    )
    features_parser.add_argument(
        '-o', '--output',
        help='Output .h5ad file (required for multiple inputs)'
    )
    features_parser.add_argument(
        '--no-binarize',
        action='store_true',
        help='Preserve original counts instead of binarizing (multi-file mode)'
    )
    features_parser.set_defaults(func=features_command)

    # Combine subcommand
    combine_parser = subparsers.add_parser(
        'combine',
        help='Merge multiple h5ad files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    combine_parser.add_argument(
        'input',
        nargs='+',
        help='Input .h5ad file(s) or glob pattern'
    )
    combine_parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output .h5ad file'
    )
    combine_parser.set_defaults(func=combine_command)

    # Metrics subcommand
    metrics_parser = subparsers.add_parser(
        'metrics',
        help='GPU-accelerated quality metrics (TSSe)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    metrics_parser.add_argument(
        'input',
        help='Input .parquet fragments file'
    )
    metrics_parser.add_argument(
        '-g', '--gtf',
        required=True,
        help='Path to GTF gene annotation file'
    )
    metrics_parser.add_argument(
        '-o', '--output',
        help='Output .csv file'
    )
    metrics_parser.add_argument(
        '--memory-resource',
        choices=['cuda-async', 'managed', 'managed-pool', 'cuda'],
        default='managed-pool',
        help='GPU memory resource: managed-pool (UVM), cuda-async (fast), cuda (basic)'
    )
    metrics_parser.add_argument(
        '--min-frags',
        type=int,
        default=100,
        help='Minimum unique fragments per cell'
    )
    metrics_parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Number of parquet row groups per batch (lower = less memory)'
    )
    metrics_parser.set_defaults(func=metrics_command)

    # Filter subcommand
    filter_parser = subparsers.add_parser(
        'filter',
        help='GPU-accelerated filtering of fragment parquet files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    filter_parser.add_argument(
        'input',
        nargs='+',
        help='Input .parquet file(s) or glob pattern'
    )
    filter_parser.add_argument(
        '-o', '--output',
        help='Output .parquet file (only for single input, default: <input>_filtered.parquet)'
    )
    filter_parser.add_argument(
        '--metrics',
        help='Path to CSV file with quality metrics (e.g., from gatac metrics)'
    )
    filter_parser.add_argument(
        '-m', '--min-fragments',
        type=int,
        default=100,
        help='Min unique fragments per cell'
    )
    filter_parser.add_argument(
        '--filter',
        dest='filter_query',
        help='Filtering query string (e.g., "tsse_score > 5 and n_unique > 1000")'
    )
    filter_parser.add_argument(
        '-g', '--genome',
        help='Genome name for chromosome filtering (e.g., hg38, mm10). Matches SnapATAC2 behavior by excluding non-standard contigs.'
    )
    filter_parser.add_argument(
        '--barcode-prefix',
        help='Prefix to add to barcodes before filtering'
    )
    filter_parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Number of parquet row groups per batch'
    )
    filter_parser.set_defaults(func=filter_command)

    # Gene subcommand
    gene_parser = subparsers.add_parser(
        'gene',
        help='Generate gene activity matrix from fragments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gene_parser.add_argument(
        'input',
        help='Input .parquet file'
    )
    gene_parser.add_argument(
        '-g', '--gtf',
        required=True,
        help='Path to GTF/GFF gene annotation file'
    )
    gene_parser.add_argument(
        '-o', '--output',
        help='Output .h5ad file'
    )
    gene_parser.add_argument(
        '--id-type',
        choices=['gene', 'transcript'],
        default='gene',
        help='Feature type to use'
    )
    gene_parser.add_argument(
        '--upstream',
        type=int,
        default=2000,
        help='Base pairs upstream of TSS'
    )
    gene_parser.add_argument(
        '--downstream',
        type=int,
        default=0,
        help='Base pairs downstream'
    )
    gene_parser.add_argument(
        '--include-gene-body',
        action='store_true',
        default=True,
        help='Include gene body in regulatory domain'
    )
    gene_parser.add_argument(
        '--no-gene-body',
        action='store_false',
        dest='include_gene_body',
        help='Exclude gene body from regulatory domain'
    )
    gene_parser.add_argument(
        '-m', '--min-fragments',
        type=int,
        default=100,
        help='Min fragments per cell'
    )
    gene_parser.add_argument(
        '-e', '--exclude-chroms',
        nargs='+',
        help='Chromosomes to exclude'
    )
    gene_parser.add_argument(
        '--metrics',
        help='Path to CSV file with quality metrics for filtering'
    )
    gene_parser.add_argument(
        '--filter',
        dest='filter_query',
        help='Filtering query string (e.g., "tsse_score > 5")'
    )
    gene_parser.add_argument(
        '--barcode-prefix',
        help='Prefix to add to barcodes'
    )
    gene_parser.add_argument(
        '--low-memory',
        action='store_true',
        help='Use low memory mode for Parquet reading'
    )
    gene_parser.set_defaults(func=gene_command)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == '__main__':
    main()
