"""
PHASE 2: MODULAR ODOR INDEX BUILDER
Builds odor_search_index.json from Pyrfume archives.
Based on pattern analysis: row index in stimuli.csv is the universal key.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

def main():
    BASE_DIR = Path("data/pyrfume_structured")
    ARCHIVES_DIR = BASE_DIR / "archives"
    PATTERN_FILE = BASE_DIR / "pattern_analysis.json"
    OUTPUT_FILE = BASE_DIR / "odor_search_index.json"
    
    print("=" * 70)
    print("MODULAR ODOR INDEX BUILDER")
    print("=" * 70)

    # Load your pattern analysis
    with open(PATTERN_FILE, 'r') as f:
        patterns = json.load(f)
    
    # STRATEGY 1: Start with HIGH-VALUE, PROVEN archives
    high_value = [
        'arctander_1960',    # Binary matrix, 2751 molecules
        'flavornet',         # Text column, reliable
        'sigma_2014',        # Binary matrix, 118 descriptors
        'leffingwell',       # Binary matrix, 112 descriptors
        'goodscents',        # Text column
        'dravnieks_1985',    # Binary matrix, 81 descriptors
        'aromadb'            # Filtered descriptors
    ]
    
    master_index = {}  # cid -> {data}
    stats = {'processed': 0, 'success': 0, 'failed': [], 'molecules_added': 0}
    
    print(f"\nProcessing {len(high_value)} high-value archives...")
    
    for archive_name in tqdm(high_value, desc="Archives"):
        archive_path = ARCHIVES_DIR / archive_name
        stats['processed'] += 1
        
        try:
            # 1. GET THE UNIVERSAL KEY: CID from stimuli.csv
            cid_list = extract_cids_from_stimuli(archive_path)
            if not cid_list:
                raise ValueError(f"No CIDs extracted from stimuli.csv")
            
            # 2. GET DESCRIPTORS using archive-specific logic
            descriptors_list = extract_descriptors_by_archive(archive_path, archive_name, len(cid_list))
            if not descriptors_list:
                print(f"  ⚠  {archive_name}: No descriptors extracted")
                continue
            
            # 3. GET MOLECULE INFO from molecules.csv
            molecules_info = load_molecule_info(archive_path)
            
            # 4. MERGE DATA (Row index is the link)
            archive_added = 0
            for i, cid in enumerate(cid_list):
                if i >= len(descriptors_list):
                    break
                    
                if not cid or pd.isna(cid):
                    continue
                
                cid = int(cid)
                descriptors = descriptors_list[i]
                
                if not descriptors:
                    continue
                
                # Find molecule details (may be by index 'i' or name lookup)
                mol_info = get_molecule_details(i, molecules_info, archive_path)
                
                if cid not in master_index:
                    master_index[cid] = {
                        'cid': cid,
                        'name': mol_info.get('name', f'CID_{cid}'),
                        'smiles': mol_info.get('smiles', ''),
                        'descriptors': descriptors,
                        'sources': [archive_name]
                    }
                    archive_added += 1
                else:
                    # Merge if already exists
                    existing = master_index[cid]
                    # Merge descriptors, deduplicate
                    combined = list(set(existing['descriptors'] + descriptors))
                    existing['descriptors'] = combined
                    if archive_name not in existing['sources']:
                        existing['sources'].append(archive_name)
            
            if archive_added > 0:
                stats['success'] += 1
                stats['molecules_added'] += archive_added
                print(f"  ✓ {archive_name}: Added {archive_added} molecules")
            else:
                stats['failed'].append(f"{archive_name}: No molecules added")
                
        except Exception as e:
            stats['failed'].append(f"{archive_name}: {str(e)[:80]}")
            print(f"  ✗ {archive_name}: {e}")
            continue
    
    # FINALIZE AND SAVE
    print(f"\nFinalizing index...")
    final_index = list(master_index.values())
    final_index.sort(key=lambda x: x['cid'])
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_index, f, indent=2)
    
    # Save CSV for inspection
    save_csv_version(final_index, BASE_DIR / "odor_search_index.csv")
    
    # REPORT
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Archives processed: {stats['processed']}")
    print(f"Archives successful: {stats['success']}")
    print(f"Total unique molecules: {len(master_index)}")
    print(f"Molecules with descriptors: {sum(1 for m in final_index if m['descriptors'])}")
    
    if stats['failed']:
        print(f"\nFailed ({len(stats['failed'])}):")
        for f in stats['failed'][:5]:
            print(f"  • {f}")
    
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"CSV for inspection: {BASE_DIR / 'odor_search_index.csv'}")
    
    # Show sample
    print(f"\nSAMPLE (first 3):")
    for item in final_index[:3]:
        desc_preview = ', '.join(item['descriptors'][:3])
        print(f"  CID {item['cid']}: {item['name'][:30]}...")
        print(f"    Sources: {item['sources']}")
        print(f"    Descriptors: {desc_preview}")

# ---------- CORE EXTRACTION MODULES ----------

def extract_cids_from_stimuli(archive_path):
    """Universal: Extract CIDs from stimuli.csv using row index as key."""
    stimuli_file = archive_path / "stimuli.csv"
    if not stimuli_file.exists():
        raise FileNotFoundError("No stimuli.csv found")
    
    df = pd.read_csv(stimuli_file)
    
    # Find CID column
    cid_col = None
    for col in df.columns:
        if 'CID' in str(col).upper():
            cid_col = col
            break
    
    if not cid_col:
        # Try to find CID in other columns
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                sample = df[col].iloc[0] if len(df) > 0 else None
                if sample and 1 < sample < 10000000:  # Reasonable CID range
                    cid_col = col
                    break
    
    if not cid_col:
        return []
    
    cids = []
    for val in df[cid_col]:
        if pd.isna(val):
            cids.append(None)
        elif isinstance(val, str) and '[' in val:
            # Handle list format like "[8051, 17750155]"
            try:
                numbers = re.findall(r'\d+', val)
                cids.append(int(numbers[0]) if numbers else None)
            except:
                cids.append(None)
        else:
            try:
                cids.append(int(val))
            except:
                cids.append(None)
    
    return cids

def extract_descriptors_by_archive(archive_path, archive_name, expected_rows):
    """Router to correct extraction method based on archive type."""
    
    # Determine behavior file
    behavior_file = None
    for bf in ['behavior_1.csv', 'behavior.csv', 'odor.csv']:
        if (archive_path / bf).exists():
            behavior_file = archive_path / bf
            break
    
    if not behavior_file:
        return []
    
    df = pd.read_csv(behavior_file)
    
    # ROUTE BASED ON KNOWN PATTERNS
    if archive_name in ['arctander_1960', 'sigma_2014', 'leffingwell', 
                       'dravnieks_1985', 'snitz_2019', 'abraham_2012']:
        return extract_binary_descriptors(df)
    
    elif archive_name in ['flavornet', 'goodscents', 'mainland_2015', 'weiss_2012']:
        return extract_text_descriptors(df)
    
    elif archive_name == 'aromadb':
        return extract_filtered_descriptors(df)
    
    else:
        # Fallback: auto-detect
        return extract_binary_descriptors(df)  # Most common

def extract_binary_descriptors(df):
    """For archives where columns ARE descriptors (acid, floral, citrus)."""
    descriptors_list = []
    
    # Identify descriptor columns (exclude metadata)
    exclude_keywords = ['stimulus', 'subject', 'participant', 'cid', 'index', 
                       'log', 'score', 'value', 'rating', 'threshold']
    
    descriptor_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in exclude_keywords):
            continue
        if len(str(col).split()) == 1 and len(str(col)) < 30:
            descriptor_cols.append(col)
    
    if not descriptor_cols:
        return []
    
    # For each row, collect columns where value indicates presence
    for _, row in df.iterrows():
        descriptors = []
        for col in descriptor_cols:
            val = row[col]
            if pd.isna(val):
                continue
            try:
                # Binary: 1, True, or any positive number
                if (isinstance(val, (int, float)) and val > 0) or \
                   (isinstance(val, str) and val.strip().lower() in ['1', 'true', 'x', '+']):
                    descriptors.append(col.lower())
            except:
                pass
        descriptors_list.append(descriptors)
    
    return descriptors_list

def extract_text_descriptors(df):
    """For archives with 'Odor' or 'Descriptors' text column."""
    descriptors_list = []
    
    # Find odor text column
    odor_col = None
    for col in df.columns:
        if 'odor' in str(col).lower() or 'descriptor' in str(col).lower():
            odor_col = col
            break
    
    if not odor_col:
        return []
    
    for _, row in df.iterrows():
        text = row[odor_col]
        if pd.isna(text):
            descriptors_list.append([])
            continue
        
        text = str(text).strip()
        if not text or text.lower() in ['na', 'nan', 'none']:
            descriptors_list.append([])
            continue
        
        # Split by common separators
        descriptors = []
        for sep in [',', ';', '/', '|', '&', ' and ', ' or ']:
            if sep in text:
                parts = [p.strip().lower() for p in text.split(sep) if p.strip()]
                descriptors.extend(parts)
                break
        
        if not descriptors:  # No separator found
            descriptors = [text.lower()]
        
        # Clean up
        clean = [d for d in descriptors if d and len(d) > 1]
        descriptors_list.append(clean)
    
    return descriptors_list

def extract_filtered_descriptors(df):
    """For aromadb with 'Filtered Descriptors' column."""
    if 'Filtered Descriptors' not in df.columns:
        return []
    
    descriptors_list = []
    for _, row in df.iterrows():
        desc = row['Filtered Descriptors']
        if pd.isna(desc):
            descriptors_list.append([])
        else:
            descriptors = [d.strip().lower() for d in str(desc).split(',')]
            descriptors_list.append([d for d in descriptors if d])
    
    return descriptors_list

def load_molecule_info(archive_path):
    """Load molecules.csv for name/SMILES lookup."""
    mol_file = archive_path / "molecules.csv"
    if not mol_file.exists():
        return {}
    
    df = pd.read_csv(mol_file)
    info_by_index = {}
    
    for i, row in df.iterrows():
        info = {
            'name': str(row.get('name', row.get('Name', ''))),
            'smiles': str(row.get('IsomericSMILES', row.get('SMILES', '')))
        }
        info_by_index[i] = info
    
    return info_by_index

def get_molecule_details(index, molecules_info, archive_path):
    """Get molecule details by index or fallback."""
    if index in molecules_info:
        return molecules_info[index]
    
    # Fallback: try to read directly
    mol_file = archive_path / "molecules.csv"
    if mol_file.exists():
        df = pd.read_csv(mol_file)
        if index < len(df):
            row = df.iloc[index]
            return {
                'name': str(row.get('name', row.get('Name', f'Molecule_{index}'))),
                'smiles': str(row.get('IsomericSMILES', row.get('SMILES', '')))
            }
    
    return {'name': f'Molecule_{index}', 'smiles': ''}

def save_csv_version(index_data, csv_path):
    """Save a CSV version for easy inspection."""
    csv_rows = []
    for item in index_data:
        csv_rows.append({
            'cid': item['cid'],
            'name': item['name'][:50],
            'smiles': item['smiles'][:100],
            'descriptors': ' | '.join(item['descriptors'][:10]),
            'descriptor_count': len(item['descriptors']),
            'sources': ', '.join(item['sources'])
        })
    
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()