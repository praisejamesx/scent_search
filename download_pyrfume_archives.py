"""
PHASE 1: PYRFUME ARCHIVE DOWNLOADER
Downloads the core triad of files from all archives.
"""
import pyrfume
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def download_targeted_files():
    """Downloads molecules.csv, stimuli.csv, and primary behavior files."""
    
    BASE_DIR = Path("data/pyrfume_structured")
    ARCHIVES_DIR = BASE_DIR / "archives"
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PYRFUME TARGETED DOWNLOADER")
    print("="*60)
    
    # Get all dataset names
    all_datasets = pyrfume.list_archives()
    print(f"Found {len(all_datasets)} datasets.\n")
    
    stats = {"total": len(all_datasets), "success": 0, "partial": 0, "failed": 0}
    manifest = {}
    
    # Core files to target
    CORE_FILES = ['molecules.csv', 'stimuli.csv', 'behavior.csv', 'behavior_1.csv']
    
    for dataset in tqdm(all_datasets, desc="Downloading"):
        dataset_dir = ARCHIVES_DIR / dataset
        dataset_dir.mkdir(exist_ok=True)
        
        dataset_files = []
        downloaded_count = 0
        
        for file_name in CORE_FILES:
            file_path = dataset_dir / file_name
            
            # Skip if already exists
            if file_path.exists():
                try:
                    # Quick validation
                    pd.read_csv(file_path, nrows=1)
                    dataset_files.append(file_name)
                    downloaded_count += 1
                    continue
                except:
                    # File is corrupted, will re-download
                    file_path.unlink(missing_ok=True)
            
            # Attempt to download
            try:
                # Strategy 1: Standard load
                df = pyrfume.load_data(f'{dataset}/{file_name}')
                success = True
            except Exception:
                try:
                    # Strategy 2: With remote=True
                    df = pyrfume.load_data(f'{dataset}/{file_name}', remote=True)
                    success = True
                except Exception as e:
                    success = False
                    # This file doesn't exist or can't be loaded; this is normal for some datasets
            
            if success:
                df.to_csv(file_path, index=False)
                dataset_files.append(file_name)
                downloaded_count += 1
        
        # Record results
        manifest[dataset] = {"local_path": str(dataset_dir), "files": dataset_files}
        
        if downloaded_count >= 2:
            stats["success"] += 1  # Good dataset: has molecules + at least one other file
        elif downloaded_count == 1:
            stats["partial"] += 1   # Only molecules.csv
        else:
            stats["failed"] += 1    # Nothing downloaded
    
    # Save the manifest for Phase 2
    manifest_path = BASE_DIR / "phase1_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Final report
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total datasets: {stats['total']}")
    print(f"✓ Good datasets (2+ files): {stats['success']}")
    print(f"⚠ Partial datasets (1 file): {stats['partial']}")
    print(f"✗ Failed datasets: {stats['failed']}")
    print(f"\nManifest saved: {manifest_path}")
    print(f"All archives saved in: {ARCHIVES_DIR}")
    
    # Preview some high-value datasets
    print(f"\nSample of high-value datasets downloaded:")
    high_value = [d for d, info in manifest.items() if 'behavior.csv' in info['files'] or 'behavior_1.csv' in info['files']]
    for dataset in high_value[:5]:
        print(f"  • {dataset}: {manifest[dataset]['files']}")
    
    return True

if __name__ == "__main__":
    download_targeted_files()