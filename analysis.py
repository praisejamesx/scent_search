"""
ANALYSIS.PY - Diagnose your Pyrfume data foundation.
Run this to understand what you have and plan Phase 2.
"""
import json
from pathlib import Path
import pandas as pd
import pyrfume
from collections import Counter
import sys

def analyze_pyrfume_base():
    """Main analysis function."""
    BASE_DIR = Path("data/pyrfume_structured")
    MANIFEST_PATH = BASE_DIR / "phase1_manifest.json"
    
    if not MANIFEST_PATH.exists():
        print("ERROR: Manifest not found. Run Phase 1 downloader first.")
        print(f"   Expected at: {MANIFEST_PATH}")
        return False
    
    print("="*70)
    print("PYRFUME DATA FOUNDATION ANALYSIS")
    print("="*70)
    
    # Load the manifest
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    # 1. BASIC STATISTICS
    print("\n1. BASIC DOWNLOAD STATISTICS")
    print("-" * 40)
    
    total_archives = len(manifest)
    archives_with_files = sum(1 for info in manifest.values() if info['files'])
    
    print(f"• Total archives in manifest: {total_archives}")
    print(f"• Archives with ≥1 downloaded file: {archives_with_files}")
    print(f"• Archives with 0 files (empty folders): {total_archives - archives_with_files}")
    
    # Count file types
    all_files = []
    for archive, info in manifest.items():
        all_files.extend(info['files'])
    
    file_counts = Counter(all_files)
    print(f"\n• Downloaded files by type:")
    for file_type, count in file_counts.most_common():
        print(f"  {file_type}: {count}")
    
    # 2. IDENTIFY SPECIFIC EMPTY FOLDERS
    print(f"\n2. EMPTY FOLDERS (NEEDS INVESTIGATION)")
    print("-" * 40)
    
    empty_archives = [archive for archive, info in manifest.items() if not info['files']]
    
    if empty_archives:
        print(f"Found {len(empty_archives)} archives with no downloaded files:")
        for archive in sorted(empty_archives):
            print(f"  • {archive}")
        
        # Check a sample of empty archives with pyrfume.show_files()
        print(f"\n3. REMOTE CHECK FOR EMPTY ARCHIVES (Sample of 3)")
        print("-" * 40)
        
        sample_size = min(3, len(empty_archives))
        for archive in empty_archives[:sample_size]:
            try:
                remote_files = pyrfume.show_files(archive)
                if remote_files:
                    print(f"✓ {archive}: Has {len(remote_files)} files remotely")
                    for fname, desc in list(remote_files.items())[:2]:
                        print(f"    - {fname}: {desc[:60]}...")
                else:
                    print(f"✗ {archive}: No files found remotely (may be metadata-only)")
            except Exception as e:
                print(f"⚠ {archive}: Error checking remote: {str(e)[:80]}")
    else:
        print("✓ No completely empty folders found!")
    
    # 3. CATEGORIZE ARCHIVES BY CONTENT TYPE
    print(f"\n4. ARCHIVE CATEGORIZATION (FOR SEARCH ENGINE)")
    print("-" * 40)
    
    # Define categories
    categories = {
        'HUMAN_ODOR_DESCRIPTORS': {
            'description': 'Primary target for search engine. Contains odor descriptor ratings.',
            'indicators': ['behavior.csv', 'behavior_1.csv', 'odor.csv'],
            'archives': []
        },
        'MOLECULES_ONLY': {
            'description': 'Has chemical data but no behavior/odor descriptors.',
            'indicators': ['molecules.csv'],
            'exclude': ['behavior.csv', 'behavior_1.csv', 'odor.csv'],
            'archives': []
        },
        'NEURAL_PHYSIOLOGY': {
            'description': 'Neural/spike data (for Hairy-Nose phase 2).',
            'indicators': ['subjects.csv', 'DeltaF', 'spike'],
            'archives': []
        },
        'MIXED_OR_UNKNOWN': {
            'description': 'Other or needs manual inspection.',
            'archives': []
        }
    }
    
    # Categorize each archive
    for archive, info in manifest.items():
        files = info['files']
        archive_path = Path(info['local_path'])
        
        # Check for human odor descriptors
        has_descriptors = any(f in files for f in ['behavior.csv', 'behavior_1.csv', 'odor.csv'])
        
        if has_descriptors:
            # Verify this is actually human odor data (not neural data)
            is_human_data = True
            
            # Quick check of behavior file content if it exists
            for behavior_file in ['behavior.csv', 'behavior_1.csv', 'odor.csv']:
                if behavior_file in files:
                    try:
                        b_path = archive_path / behavior_file
                        df = pd.read_csv(b_path, nrows=5)
                        
                        # Check if columns look like odor descriptors
                        # (not neural/spike data columns)
                        neural_indicators = ['DeltaF', 'spike', 'glom', 'neuron', 'firing']
                        col_str = str(df.columns).lower()
                        if any(indicator in col_str for indicator in neural_indicators):
                            is_human_data = False
                            break
                            
                    except Exception as e:
                        # Can't read file, mark for manual check
                        pass
            
            if is_human_data:
                categories['HUMAN_ODOR_DESCRIPTORS']['archives'].append(archive)
            else:
                categories['NEURAL_PHYSIOLOGY']['archives'].append(archive)
                
        elif 'molecules.csv' in files:
            categories['MOLECULES_ONLY']['archives'].append(archive)
        else:
            categories['MIXED_OR_UNKNOWN']['archives'].append(archive)
    
    # Print categorization results
    for category, data in categories.items():
        count = len(data['archives'])
        if count > 0:
            print(f"\n• {category}: {count} archives")
            print(f"  {data['description']}")
            if 'archives' in data and data['archives']:
                print(f"  Sample: {', '.join(sorted(data['archives'])[:3])}")
                if len(data['archives']) > 3:
                    print(f"  (and {len(data['archives']) - 3} more)")
    
    # 4. PRIORITY LIST FOR PHASE 2
    print(f"\n5. PHASE 2 PRIORITY LIST")
    print("-" * 40)
    
    priority_archives = categories['HUMAN_ODOR_DESCRIPTORS']['archives']
    
    if priority_archives:
        print("✓ HIGHEST PRIORITY (Start with these for search engine):")
        for archive in sorted(priority_archives):
            print(f"  • {archive}")
            
        # Check data quality for top 3 priority archives
        print(f"\n✓ DATA QUALITY CHECK (Top 3 archives):")
        for archive in sorted(priority_archives)[:3]:
            archive_info = manifest[archive]
            print(f"\n  {archive}:")
            print(f"    Files: {', '.join(archive_info['files'])}")
            
            # Count molecules and descriptors if possible
            try:
                mol_path = Path(archive_info['local_path']) / 'molecules.csv'
                if mol_path.exists():
                    mol_df = pd.read_csv(mol_path)
                    print(f"    Molecules: {len(mol_df)}")
                
                # Check behavior file
                for b_file in ['behavior.csv', 'behavior_1.csv', 'odor.csv']:
                    b_path = Path(archive_info['local_path']) / b_file
                    if b_path.exists():
                        b_df = pd.read_csv(b_path, nrows=0)  # Just headers
                        print(f"    {b_file} columns: {len(b_df.columns)}")
                        # Show first few column names if they look like descriptors
                        descriptor_cols = [c for c in b_df.columns if c not in 
                                          ['CID', 'Stimulus', 'Subject', 'Index']]
                        if descriptor_cols:
                            print(f"      Sample columns: {descriptor_cols[:3]}")
                        break
            except Exception as e:
                print(f"    Could not inspect files: {str(e)[:60]}")
    else:
        print("⚠ WARNING: No high-priority archives found!")
        print("  You may need to fix downloads or adjust categorization logic.")
    
    # 5. ACTION PLAN
    print(f"\n6. RECOMMENDED ACTION PLAN")
    print("-" * 40)
    
    print("1. START BUILDING SEARCH ENGINE WITH:")
    if priority_archives:
        print(f"   • Begin with: {priority_archives[0]}")
        print(f"   • Then add: {priority_archives[1] if len(priority_archives) > 1 else 'N/A'}")
    else:
        print("   • No clear starting point. Re-run Phase 1 downloader.")
    
    print(f"\n2. FIX EMPTY FOLDERS LATER:")
    print(f"   • {len(empty_archives)} archives need investigation")
    if empty_archives:
        print(f"   • Start with: {empty_archives[0]}")
    
    print(f"\n3. PHASE 2 SCRIPT SHOULD:")
    print(f"   • Process archives in priority order")
    print(f"   • Skip archives with missing files")
    print(f"   • Build odor_search_index.json incrementally")
    
    # Save detailed report
    report = {
        'summary': {
            'total_archives': total_archives,
            'archives_with_files': archives_with_files,
            'empty_archives': empty_archives,
            'file_counts': dict(file_counts)
        },
        'categorization': {
            cat: {'count': len(data['archives']), 'archives': data['archives']}
            for cat, data in categories.items()
        },
        'priority_archives': priority_archives,
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    report_path = BASE_DIR / "analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n" + "="*70)
    print(f"ANALYSIS COMPLETE")
    print(f"="*70)
    print(f"Full report saved: {report_path}")
    print(f"Next: Start building phase2_index_builder.py")
    
    if priority_archives:
        print(f"\nRecommended first command for Phase 2:")
        print(f"  python phase2_index_builder.py --archive {priority_archives[0]}")
    
    return True

if __name__ == "__main__":
    # Add the current directory to path for imports
    sys.path.append('.')
    analyze_pyrfume_base()