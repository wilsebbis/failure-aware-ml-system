#!/usr/bin/env python3
"""
Dataset Download Script

Downloads datasets from Kaggle for the Failure-Aware ML System.
Uses the Python kaggle library directly (no CLI needed).

Setup:
    Create .env file in project root with:
    KAGGLE_API_TOKEN="your_token_here"

Usage:
    python3 scripts/download_data.py --dataset all
    python3 scripts/download_data.py --dataset home_credit
    python3 scripts/download_data.py --dataset ieee_cis
    python3 scripts/download_data.py --dataset lending_club
    python3 scripts/download_data.py --dataset uci_credit
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# Dataset configurations
DATASETS = {
    "uci_credit": {
        "type": "dataset",
        "kaggle_id": "uciml/default-of-credit-card-clients-dataset",
        "output_dir": "data/raw",
        "rename_to": "UCI_Credit_Card.csv",
        "description": "UCI Credit Card Default (single CSV, ~3MB)",
    },
    "home_credit": {
        "type": "competition",
        "kaggle_id": "home-credit-default-risk",
        "output_dir": "data/raw/home_credit",
        "description": "Home Credit Default Risk (7 tables, ~800MB)",
    },
    "ieee_cis": {
        "type": "competition", 
        "kaggle_id": "ieee-fraud-detection",
        "output_dir": "data/raw/ieee_cis",
        "description": "IEEE-CIS Fraud Detection (500k+ rows, ~1.2GB)",
    },
    "lending_club": {
        "type": "dataset",
        "kaggle_id": "wordsforthewise/lending-club",
        "output_dir": "data/raw/lending_club",
        "description": "Lending Club Loan Data (2007-2020, ~1.5GB)",
    },
}


def load_env():
    """Load .env file if it exists."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip('"\'')
                    os.environ[key] = value


def check_kaggle_api():
    """Check if Kaggle API is configured and library is available."""
    # Load .env first
    load_env()
    
    # Check for credentials
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    has_user_key = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    
    if has_token:
        print("✓ Using KAGGLE_API_TOKEN from environment")
    elif has_user_key:
        print("✓ Using KAGGLE_USERNAME/KAGGLE_KEY from environment")
    elif has_json:
        print("✓ Using ~/.kaggle/kaggle.json")
    else:
        print("❌ Kaggle API not configured!")
        print("\nOption 1: Create .env file in project root with:")
        print('  KAGGLE_API_TOKEN="your_token_here"')
        print("\nOption 2: Set username/key in .env:")
        print('  KAGGLE_USERNAME="your_username"')
        print('  KAGGLE_KEY="your_api_key"')
        print("\nOption 3: Traditional method:")
        print("  1. Go to kaggle.com/settings")
        print("  2. API section → Create New Token")
        print("  3. Move kaggle.json to ~/.kaggle/")
        return False
    
    # Check if kaggle library is available
    try:
        import kaggle
        return True
    except ImportError:
        print("\n❌ Kaggle library not installed!")
        print("Install with: pip install kaggle")
        return False


def download_dataset(name: str, config: dict, project_root: Path):
    """Download a single dataset using Python API."""
    from kaggle import api as kaggle_api
    
    output_dir = project_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading {name}...")
    print(f"   {config['description']}")
    print(f"   Target: {output_dir}")
    
    try:
        if config["type"] == "competition":
            kaggle_api.competition_download_files(
                config["kaggle_id"],
                path=str(output_dir),
                quiet=False
            )
        else:
            kaggle_api.dataset_download_files(
                config["kaggle_id"],
                path=str(output_dir),
                unzip=True,
                quiet=False
            )
        
        # Unzip competition files (they come as zip)
        zip_files = list(output_dir.glob("*.zip"))
        for zip_file in zip_files:
            print(f"   📦 Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(output_dir)
            zip_file.unlink()  # Remove zip after extraction
        
        # Handle rename for UCI dataset
        if "rename_to" in config:
            downloaded = list(output_dir.glob("*.csv"))
            if downloaded and len(downloaded) == 1:
                target_name = output_dir / config["rename_to"]
                if not target_name.exists():
                    downloaded[0].rename(target_name)
        
        print(f"   ✅ Downloaded successfully")
        return True
            
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Error: {error_msg}")
        
        if "403" in error_msg or "must accept" in error_msg.lower():
            print(f"   → Accept competition rules at: https://kaggle.com/c/{config['kaggle_id']}")
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Failure-Aware ML System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 scripts/download_data.py --dataset all
    python3 scripts/download_data.py --dataset home_credit
    python3 scripts/download_data.py --list
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        default="all",
        help="Dataset to download (or 'all')"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable datasets:")
        for name, config in DATASETS.items():
            print(f"  {name:15} - {config['description']}")
        return
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    # Check Kaggle API
    if not check_kaggle_api():
        sys.exit(1)
    
    # Determine which datasets to download
    if args.dataset == "all":
        datasets_to_download = list(DATASETS.keys())
    elif args.dataset in DATASETS:
        datasets_to_download = [args.dataset]
    else:
        print(f"❌ Unknown dataset: {args.dataset}")
        print(f"   Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)
    
    print("=" * 60)
    print("FAILURE-AWARE ML SYSTEM - Dataset Downloader")
    print("=" * 60)
    
    results = {}
    for name in datasets_to_download:
        config = DATASETS[name]
        results[name] = download_dataset(name, config, project_root)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    if all(results.values()):
        print("\n✅ All datasets ready!")
        print("\nRun the pipeline:")
        print("  uv run python -m src.main --dataset home_credit")
    else:
        print("\n⚠️  Some downloads failed. Check the errors above.")


if __name__ == "__main__":
    main()
