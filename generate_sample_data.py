"""
Generate sample stellar light curve data for testing and demonstration.

This script creates synthetic light curves with various types of anomalies:
- Exoplanet transits
- Stellar flares
- Random outliers
- Normal stellar variability
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def generate_lightcurve(n_points=2000, has_transit=False, has_flare=False,
                       has_outliers=False, noise_level=5.0, seed=None):
    """
    Generate a synthetic stellar light curve.

    Args:
        n_points: Number of data points
        has_transit: Whether to include exoplanet transit
        has_flare: Whether to include stellar flare
        has_outliers: Whether to include random outliers
        noise_level: Standard deviation of measurement noise
        seed: Random seed for reproducibility

    Returns:
        DataFrame with time, flux, flux_err columns
    """
    if seed is not None:
        np.random.seed(seed)

    # Time array (in days)
    time = np.linspace(0, 100, n_points)

    # Base flux (normalized to 1000)
    base_flux = 1000.0
    flux = np.ones(n_points) * base_flux

    # Add stellar variability (slow variations)
    # Combination of sine waves at different periods
    variability = (
        10 * np.sin(2 * np.pi * time / 50) +  # 50-day period
        5 * np.sin(2 * np.pi * time / 20) +   # 20-day period
        3 * np.sin(2 * np.pi * time / 10)     # 10-day period
    )
    flux += variability

    # Add measurement noise
    flux += np.random.normal(0, noise_level, n_points)

    # Add exoplanet transit
    if has_transit:
        # Transit parameters
        period = 15.0  # days
        duration = 0.15  # days (about 3.6 hours)
        depth = 30.0  # flux units

        # Add multiple transits
        n_transits = int(time[-1] / period)
        for i in range(n_transits):
            transit_center = period * (i + 0.5)

            # Create transit shape (simplified box model with ingress/egress)
            for j, t in enumerate(time):
                if abs(t - transit_center) < duration / 2:
                    # Distance from transit center (normalized)
                    phase = abs(t - transit_center) / (duration / 2)

                    if phase < 0.2:  # Ingress
                        flux[j] -= depth * (phase / 0.2)
                    elif phase > 0.8:  # Egress
                        flux[j] -= depth * ((1 - phase) / 0.2)
                    else:  # Full transit
                        flux[j] -= depth

    # Add stellar flare
    if has_flare:
        # Flare parameters
        n_flares = np.random.randint(2, 5)

        for _ in range(n_flares):
            flare_time = np.random.uniform(20, 80)  # Random time
            flare_duration = np.random.uniform(0.1, 0.5)  # days
            flare_amplitude = np.random.uniform(80, 200)

            # Exponential rise and decay
            for j, t in enumerate(time):
                if t >= flare_time and t < flare_time + flare_duration:
                    phase = (t - flare_time) / flare_duration

                    # Fast rise, slow decay
                    if phase < 0.2:
                        intensity = (phase / 0.2)
                    else:
                        intensity = np.exp(-5 * (phase - 0.2))

                    flux[j] += flare_amplitude * intensity

    # Add random outliers
    if has_outliers:
        n_outliers = np.random.randint(10, 30)
        outlier_indices = np.random.choice(n_points, n_outliers, replace=False)

        for idx in outlier_indices:
            # Random spike or dip
            outlier_value = np.random.choice([-1, 1]) * np.random.uniform(50, 150)
            flux[idx] += outlier_value

    # Generate flux errors (photon noise model)
    # Error scales with square root of flux
    flux_err = np.sqrt(np.abs(flux)) * 0.1 + np.random.uniform(1, 3, n_points)

    return pd.DataFrame({
        'time': time,
        'flux': flux,
        'flux_err': flux_err
    })


def save_lightcurve(df, filename, format='csv'):
    """
    Save light curve to file.

    Args:
        df: DataFrame with time, flux, flux_err
        filename: Output filename
        format: 'csv' or 'fits'
    """
    if format == 'csv':
        df.to_csv(filename, index=False)
        print(f"Saved CSV: {filename}")

    elif format == 'fits':
        from astropy.io import fits
        from astropy.table import Table

        # Convert DataFrame to Astropy Table
        table = Table.from_pandas(df)

        # Create FITS file
        primary_hdu = fits.PrimaryHDU()
        table_hdu = fits.BinTableHDU(table, name='LIGHTCURVE')

        # Add metadata to primary header
        primary_hdu.header['OBJECT'] = 'Simulated Star'
        primary_hdu.header['TELESCOP'] = 'Synthetic'
        primary_hdu.header['INSTRUME'] = 'Generator'

        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(filename, overwrite=True)
        print(f"Saved FITS: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample stellar light curve data')
    parser.add_argument('--output-dir', type=str, default='data/samples',
                       help='Output directory for generated files')
    parser.add_argument('--format', type=str, choices=['csv', 'fits', 'both'],
                       default='both', help='Output format')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of light curves to generate')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Stellar Light Curve Sample Data Generator")
    print("="*60)
    print()

    # Generate different types of light curves
    samples = [
        {
            'name': 'normal_star',
            'description': 'Normal star with only stellar variability',
            'params': {'has_transit': False, 'has_flare': False, 'has_outliers': False, 'seed': 42}
        },
        {
            'name': 'exoplanet_transit',
            'description': 'Star with periodic exoplanet transits',
            'params': {'has_transit': True, 'has_flare': False, 'has_outliers': False, 'seed': 123}
        },
        {
            'name': 'stellar_flares',
            'description': 'Active star with multiple flares',
            'params': {'has_transit': False, 'has_flare': True, 'has_outliers': False, 'seed': 456}
        },
        {
            'name': 'noisy_outliers',
            'description': 'Star with instrumental artifacts and outliers',
            'params': {'has_transit': False, 'has_flare': False, 'has_outliers': True, 'seed': 789}
        },
        {
            'name': 'complex_system',
            'description': 'Complex system with transits, flares, and outliers',
            'params': {'has_transit': True, 'has_flare': True, 'has_outliers': True, 'seed': 999}
        },
    ]

    for i, sample in enumerate(samples[:args.n_samples]):
        print(f"\nGenerating {i+1}/{min(args.n_samples, len(samples))}: {sample['name']}")
        print(f"  Description: {sample['description']}")

        # Generate light curve
        df = generate_lightcurve(n_points=2000, **sample['params'])

        # Save in requested format(s)
        if args.format in ['csv', 'both']:
            csv_file = output_dir / f"{sample['name']}.csv"
            save_lightcurve(df, csv_file, format='csv')

        if args.format in ['fits', 'both']:
            fits_file = output_dir / f"{sample['name']}.fits"
            save_lightcurve(df, fits_file, format='fits')

        # Print statistics
        print(f"  Points: {len(df)}")
        print(f"  Time range: {df['time'].min():.2f} - {df['time'].max():.2f} days")
        print(f"  Flux mean: {df['flux'].mean():.2f}")
        print(f"  Flux std: {df['flux'].std():.2f}")

    print()
    print("="*60)
    print(f"✅ Generated {min(args.n_samples, len(samples))} sample light curves")
    print(f"📁 Saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == '__main__':
    main()
