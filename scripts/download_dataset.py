#!/usr/bin/env python3
"""
CLI script to download and label Kepler/TESS light curves for the dataset.

Usage:
    python scripts/download_dataset.py --mission Kepler --n-planets 10 --n-normal 5
    python scripts/download_dataset.py --mission TESS --n-planets 50 --n-normal 30
    python scripts/download_dataset.py --smoke-test  # Quick 5+5 download
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.acquisition import MastDataAcquisitor


def main():
    parser = argparse.ArgumentParser(
        description='Download labeled Kepler/TESS light curves from MAST'
    )
    parser.add_argument('--mission', type=str, default='Kepler',
                        choices=['Kepler', 'TESS'],
                        help='Mission to download from (default: Kepler)')
    parser.add_argument('--n-planets', type=int, default=150,
                        help='Number of planet-hosting stars (default: 150)')
    parser.add_argument('--n-normal', type=int, default=100,
                        help='Number of non-planet stars (default: 100)')
    parser.add_argument('--output-dir', type=str, default='data/labeled',
                        help='Output directory (default: data/labeled)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between downloads in seconds (default: 1.0)')
    parser.add_argument('--max-quarters', type=int, default=None,
                        help='Max quarters/sectors per target (default: all, smoke-test: 3)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick smoke test: download 5 planets + 5 normal (3 quarters each)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    if args.smoke_test:
        args.n_planets = 5
        args.n_normal = 5
        if args.max_quarters is None:
            args.max_quarters = 3
        print(f"=== SMOKE TEST MODE: 5 planets + 5 normal ({args.max_quarters} quarters each) ===\n")

    print(f"Mission:        {args.mission}")
    print(f"Planet hosts:   {args.n_planets}")
    print(f"Normal stars:   {args.n_normal}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Delay:          {args.delay}s")
    print()

    acquisitor = MastDataAcquisitor(output_dir=args.output_dir)

    metadata = acquisitor.build_dataset(
        n_planet_hosts=args.n_planets,
        n_non_planet=args.n_normal,
        mission=args.mission,
        delay_between=args.delay,
        max_quarters=args.max_quarters,
    )

    print(f"\nDataset saved to {args.output_dir}/")
    print(f"Total targets: {len(metadata)}")
    if len(metadata) > 0:
        print(f"Planet hosts:  {(metadata['label_type'] == 'transit').sum()}")
        print(f"Normal stars:  {(metadata['label_type'] == 'normal').sum()}")
        print(f"Total points:  {metadata['n_points'].sum():,}")


if __name__ == '__main__':
    main()
