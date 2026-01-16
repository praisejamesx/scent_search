"""
PYRFUME COMPREHENSIVE ANALYSIS - FIXED VERSION
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter
import sys

def comprehensive_analysis_fixed():
    BASE_DIR = Path("data/pyrfume_structured")
    
    print("="*80)
    print("PYRFUME STRUCTURE ANALYSIS - PATTERN DISCOVERY")
    print("="*80)
    
    # Load priority archives
    with open(BASE_DIR / "analysis_report.json", 'r') as f:
        analysis_report = json.load(f)
    
    priority_archives = analysis_report['categorization']['HUMAN_ODOR_DESCRIPTORS']['archives']
    
    print(f"\nAnalyzing patterns across {len(priority_archives)} archives...\n")
    
    archive_patterns = {}
    
    for archive_name in priority_archives:
        archive_path = BASE_DIR / "archives" / archive_name
        
        pattern = {
            'archive': archive_name,
            'stimuli_has_cid': False,
            'behavior_format': None,
            'descriptor_columns': [],
            'notes': []
        }
        
        # Check stimuli.csv
        stimuli_path = archive_path / "stimuli.csv"
        if stimuli_path.exists():
            try:
                stimuli_df = pd.read_csv(stimuli_path, nrows=5)
                pattern['stimuli_columns'] = list(stimuli_df.columns)
                
                # Check for CID column
                cid_col = None
                for col in stimuli_df.columns:
                    if 'CID' in str(col).upper():
                        cid_col = col
                        break
                
                if cid_col:
                    pattern['stimuli_has_cid'] = True
                    pattern['cid_column'] = cid_col
                    
                    # Check if CIDs are lists (like "[8051, 17750155]")
                    sample_cid = stimuli_df[cid_col].iloc[0] if len(stimuli_df) > 0 else None
                    if isinstance(sample_cid, str) and '[' in sample_cid:
                        pattern['notes'].append('CIDs are lists/arrays')
                
                # Check for other identifier columns
                for col in stimuli_df.columns:
                    col_lower = str(col).lower()
                    if any(keyword in col_lower for keyword in ['name', 'cas', 'odorant', 'chemical']):
                        pattern['notes'].append(f'Has identifier column: {col}')
                        
            except Exception as e:
                pattern['notes'].append(f'Error reading stimuli.csv: {str(e)[:50]}')
        
        # Check behavior files
        for b_file in ['behavior.csv', 'behavior_1.csv', 'behavior_2.csv', 'odor.csv']:
            b_path = archive_path / b_file
            if b_path.exists():
                try:
                    b_df = pd.read_csv(b_path, nrows=10)
                    pattern['behavior_file'] = b_file
                    pattern['behavior_columns'] = list(b_df.columns)
                    
                    # Determine format
                    cols_lower = [str(col).lower() for col in b_df.columns]
                    
                    # Check for binary descriptor columns (like arctander_1960)
                    descriptor_cols = []
                    for col in b_df.columns:
                        col_str = str(col)
                        # Skip metadata columns
                        if any(meta in col_str.lower() for meta in ['stimulus', 'subject', 'cid', 'index', 'log', 'score', 'value', 'rating', 'participant']):
                            continue
                        # Likely descriptor if single word and not too long
                        if len(col_str.split()) == 1 and len(col_str) < 30:
                            descriptor_cols.append(col_str)
                    
                    if len(descriptor_cols) > 10:  # Many descriptor columns
                        pattern['behavior_format'] = 'binary_descriptor_matrix'
                        pattern['descriptor_columns'] = descriptor_cols[:20]  # Just first 20
                        pattern['notes'].append(f'Binary descriptors: {len(descriptor_cols)} columns')
                    
                    # Check for odor text column
                    elif 'Odor' in b_df.columns or 'odor' in b_df.columns or 'Descriptors' in b_df.columns:
                        pattern['behavior_format'] = 'odor_text_column'
                        odor_col = 'Odor' if 'Odor' in b_df.columns else ('odor' if 'odor' in b_df.columns else 'Descriptors')
                        pattern['notes'].append(f'Has {odor_col} text column')
                    
                    # Check for filtered descriptors (aromadb)
                    elif 'Filtered Descriptors' in b_df.columns:
                        pattern['behavior_format'] = 'filtered_descriptors'
                        pattern['notes'].append('Has Filtered Descriptors column')
                    
                    else:
                        pattern['behavior_format'] = 'unknown_or_metadata'
                        pattern['notes'].append('No obvious descriptor columns found')
                        
                    break  # Only process first behavior file
                    
                except Exception as e:
                    pattern['notes'].append(f'Error reading {b_file}: {str(e)[:50]}')
        
        archive_patterns[archive_name] = pattern
    
    # Analyze patterns
    print("\nPATTERN SUMMARY")
    print("-" * 60)
    
    stimuli_with_cid = sum(1 for p in archive_patterns.values() if p.get('stimuli_has_cid', False))
    print(f"Archives with CID in stimuli.csv: {stimuli_with_cid}/{len(archive_patterns)}")
    
    behavior_formats = Counter([p.get('behavior_format', 'unknown') for p in archive_patterns.values()])
    print(f"\nBehavior file formats:")
    for format_name, count in behavior_formats.most_common():
        print(f"  {format_name}: {count}")
    
    # Group archives by extraction strategy
    print("\nEXTRACTION STRATEGIES BY ARCHIVE TYPE")
    print("-" * 60)
    
    strategies = {
        'binary_descriptor_matrix': [],
        'odor_text_column': [],
        'filtered_descriptors': [],
        'metadata_only': [],
        'needs_inspection': []
    }
    
    for archive_name, pattern in archive_patterns.items():
        behavior_format = pattern.get('behavior_format', 'unknown')
        
        if behavior_format == 'binary_descriptor_matrix':
            strategies['binary_descriptor_matrix'].append(archive_name)
        elif behavior_format == 'odor_text_column':
            strategies['odor_text_column'].append(archive_name)
        elif behavior_format == 'filtered_descriptors':
            strategies['filtered_descriptors'].append(archive_name)
        elif behavior_format == 'unknown_or_metadata':
            # Check if it has any descriptor-like columns
            if pattern.get('descriptor_columns'):
                strategies['binary_descriptor_matrix'].append(archive_name)
            else:
                strategies['metadata_only'].append(archive_name)
        else:
            strategies['needs_inspection'].append(archive_name)
    
    for strategy_type, archives in strategies.items():
        if archives:
            print(f"\n{strategy_type.upper().replace('_', ' ')} ({len(archives)}):")
            for archive in sorted(archives)[:10]:  # Show first 10
                print(f"  • {archive}")
            if len(archives) > 10:
                print(f"  • ... and {len(archives) - 10} more")
    
    # Create extraction blueprint
    extraction_blueprint = {}
    
    for archive_name, pattern in archive_patterns.items():
        blueprint = {
            'stimuli_has_cid': pattern.get('stimuli_has_cid', False),
            'behavior_format': pattern.get('behavior_format', 'unknown'),
            'extraction_method': None,
            'steps': []
        }
        
        if pattern.get('stimuli_has_cid'):
            blueprint['steps'].append(f"Read CID from stimuli.csv column: {pattern.get('cid_column', 'CID')}")
            blueprint['steps'].append("Map row index in behavior file to row in stimuli.csv")
        else:
            blueprint['steps'].append("Need alternative linking method (by name or CAS)")
        
        if pattern.get('behavior_format') == 'binary_descriptor_matrix':
            blueprint['extraction_method'] = 'extract_binary_descriptors'
            blueprint['steps'].append(f"Extract descriptors from {len(pattern.get('descriptor_columns', []))} binary columns")
            blueprint['steps'].append("Descriptor is present if value > 0 or == 1")
        
        elif pattern.get('behavior_format') == 'odor_text_column':
            blueprint['extraction_method'] = 'extract_text_descriptors'
            blueprint['steps'].append("Parse comma-separated odor descriptors from text column")
        
        elif pattern.get('behavior_format') == 'filtered_descriptors':
            blueprint['extraction_method'] = 'extract_filtered_descriptors'
            blueprint['steps'].append("Use 'Filtered Descriptors' column")
        
        else:
            blueprint['extraction_method'] = 'needs_inspection'
            blueprint['steps'].append("Manual inspection required")
        
        extraction_blueprint[archive_name] = blueprint
    
    # Save results
    output_data = {
        'archive_patterns': archive_patterns,
        'strategies_summary': strategies,
        'extraction_blueprint': extraction_blueprint,
        'summary': {
            'total_archives': len(priority_archives),
            'archives_with_cid_in_stimuli': stimuli_with_cid,
            'behavior_format_counts': dict(behavior_formats)
        }
    }
    
    with open(BASE_DIR / "pattern_analysis.json", 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nAnalysis saved to: {BASE_DIR / 'pattern_analysis.json'}")
    
    # Create actionable plan
    print("\nACTIONABLE PLAN FOR INDEX BUILDER")
    print("="*60)
    
    print("\n1. START WITH THESE ARCHIVES (Easiest to extract):")
    easy_archives = strategies.get('binary_descriptor_matrix', [])[:5]
    for archive in easy_archives:
        print(f"   • {archive}")
    
    print("\n2. EXTRACTION METHODOLOGY:")
    print("   a. For archives with CID in stimuli.csv:")
    print("      - Read stimuli.csv to get CID for each row")
    print("      - Map behavior row index to stimuli row index")
    print("      - Extract descriptors from behavior columns")
    print("      - Look up molecule info from molecules.csv by name or SMILES")
    
    print("\n   b. For binary descriptor archives (arctander_1960, sigma_2014, etc.):")
    print("      - Each column is a descriptor (acid, floral, citrus)")
    print("      - Value 1 or >0 means the descriptor applies")
    
    print("\n   c. For text descriptor archives (flavornet, goodscents):")
    print("      - Parse comma-separated descriptors from text column")
    
    print("\n3. PRIORITY ORDER:")
    print("   1. arctander_1960 (2751 molecules, 76 descriptors)")
    print("   2. flavornet (text descriptors)")
    print("   3. sigma_2014 (118 descriptor columns)")
    print("   4. leffingwell (112 descriptor columns)")
    print("   5. dravnieks_1985 (81 descriptor columns)")
    
    return output_data

if __name__ == "__main__":
    comprehensive_analysis_fixed()