"""
GATAC Command Line Interface.

Usage:
    gatac convert <input.tsv.gz> [output.parquet]
    gatac tile <input.parquet> [-o output] [-t tile_size] [-m min_frags]
    gatac features <input.h5ad> [-n n_features] [-o output]
    gatac metrics <input.parquet> -g <annotations.gtf> [-o output]
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
    from .convert import make_parquet

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path = args.output if args.output else None
    make_parquet(input_path, output_path, progress=args.progress)


def tile_command(args):
    """Handle 'gatac tile' subcommand."""
    from .process import make_tile_matrix

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
    from .features import select_features, select_features_multi

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

    if args.streaming:
        # Use Polars GPU streaming for out-of-core processing
        from .metrics_streaming import load_tss_from_gtf_polars, compute_metrics_streaming
        
        logging.info(f"Loading fragments from {input_path} (streaming mode)")
        tss_lf = load_tss_from_gtf_polars(gtf_path)
        results = compute_metrics_streaming(
            input_path,
            tss_lf,
            engine_mode="streaming",
            memory_resource=args.memory_resource,
            min_unique_frags=args.min_frags,
            batch_row_groups=args.batch_size,
        )
        
        logging.info(f"Saving results to {output_path}")
        results.write_csv(str(output_path))
        logging.info(f"Successfully processed {len(results):,} cells.")
    else:
        # Use existing cuDF implementation (default)
        from .metrics import load_tss_from_gtf, compute_metrics
        from .process import read_fragments_parquet

        logging.info(f"Loading fragments from {input_path}")
        # Use optimized reader with specific dtypes to save GPU memory
        fragments = read_fragments_parquet(input_path, low_memory=True)
        
        tss_df = load_tss_from_gtf(gtf_path)
        results = compute_metrics(fragments, tss_df)
        
        logging.info(f"Saving results to {output_path}")
        results.to_csv(str(output_path), index=False)
        logging.info(f"Successfully processed {len(results):,} cells.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='gatac',
        description='GPU-accelerated ATAC-seq processing toolkit'
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
        help='Convert ATAC fragments TSV.GZ to Parquet'
    )
    convert_parser.add_argument(
        'input',
        help='Input .tsv.gz file'
    )
    convert_parser.add_argument(
        'output',
        nargs='?',
        help='Output .parquet file (default: input name with .parquet)'
    )
    convert_parser.add_argument(
        '-p', '--progress',
        action='store_true',
        help='Show progress bar'
    )
    convert_parser.set_defaults(func=convert_command)

    # Tile subcommand
    tile_parser = subparsers.add_parser(
        'tile',
        help='Process fragments to tile matrix'
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
        help='Tile size in bp (default: 5000)'
    )
    tile_parser.add_argument(
        '-m', '--min-fragments',
        type=int,
        default=100,
        help='Min fragments per cell (default: 100)'
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
        help='Chromosomes to exclude (default: chrM chrY M Y)'
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
    tile_parser.set_defaults(func=tile_command)

    # Features subcommand
    features_parser = subparsers.add_parser(
        'features',
        help='GPU-accelerated feature selection'
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
        help='Number of features to select (default: 500000)'
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

    # Metrics subcommand
    metrics_parser = subparsers.add_parser(
        'metrics',
        help='GPU-accelerated quality metrics (TSSe)'
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
        '--streaming',
        action='store_true',
        help='Use Polars GPU streaming for out-of-core processing (larger than VRAM datasets)'
    )
    metrics_parser.add_argument(
        '--memory-resource',
        choices=['cuda-async', 'managed', 'managed-pool', 'cuda'],
        default='managed-pool',
        help='GPU memory resource: managed-pool (UVM, default), cuda-async (fast), cuda (basic)'
    )
    metrics_parser.add_argument(
        '--min-frags',
        type=int,
        default=100,
        help='Minimum unique fragments per cell (default: 100)'
    )
    metrics_parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of parquet row groups per batch (default: 5, lower = less memory)'
    )
    metrics_parser.set_defaults(func=metrics_command)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == '__main__':
    main()
