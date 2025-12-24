import pyrfume
import pandas as pd
import numpy as np
from pathlib import Path
import json

def explore_pyrfume_data():
    """Explore available datasets in Pyrfume"""
    print("Available datasets in Pyrfume:")
    print("-" * 50)
    
    # List available datasets
    datasets = [
        'goodscents',      # The Good Scents Company data
        'leffingwell',     # Leffingwell odor data
        'flavor',          # Flavor data
        'sigma_ffn',       # Sigma FFN data
    ]
    
    for dataset in datasets:
        try:
            data = pyrfume.load_data(dataset)
            print(f"\n{dataset}:")
            print(f"  Type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())}")
                for key, value in data.items():
                    if hasattr(value, 'shape'):
                        print(f"    {key}: {value.shape}")
            elif hasattr(data, 'shape'):
                print(f"  Shape: {data.shape}")
        except Exception as e:
            print(f"  Error loading {dataset}: {e}")
    
    return datasets

def load_goodscents_data():
    """Load and process The Good Scents Company data"""
    print("\nLoading The Good Scents Company data...")
    
    try:
        # Load the dataset
        data = pyrfume.load_data('goodscents')
        
        print("\nDataset structure:")
        if isinstance(data, dict):
            for key, df in data.items():
                print(f"\n{key}:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)[:10]}...")
                
                # Show sample for main datasets
                if key in ['molecules', 'odor', 'behavior']:
                    print(f"\n  Sample rows:")
                    print(df.head(3).to_string())
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try alternative approach
        try:
            from pyrfume.datasets import GOOD_SCENTS
            data = GOOD_SCENTS.load()
            return data
        except:
            print("Could not load Good Scents data")
            return None

def create_initial_dataset():
    """Create a cleaned dataset for initial development"""
    data = load_goodscents_data()
    
    if data is None:
        print("Creating synthetic dataset for development...")
        return create_synthetic_dataset()
    
    # Extract and combine relevant data
    molecules = []
    descriptors = []
    
    if 'molecules' in data and 'odor' in data:
        # Process molecules
        mol_df = data['molecules'].copy()
        odor_df = data['odor'].copy()
        
        # Clean data
        mol_df = mol_df.dropna(subset=['SMILES', 'CID'])
        odor_df = odor_df.dropna(subset=['Odor'])
        
        # Create unified records
        for idx, row in mol_df.iterrows():
            cid = row.get('CID')
            if pd.notna(cid):
                # Find odor descriptors for this molecule
                molecule_descriptors = odor_df[odor_df['CID'] == cid]['Odor']
                if not molecule_descriptors.empty:
                    # Combine all descriptors for this molecule
                    desc_list = []
                    for desc in molecule_descriptors:
                        if pd.notna(desc):
                            # Clean and split descriptors
                            clean_desc = str(desc).lower().strip()
                            # Split by common separators
                            for sep in [',', ';', 'and', 'with']:
                                clean_desc = clean_desc.replace(sep, '|')
                            parts = [p.strip() for p in clean_desc.split('|') if p.strip()]
                            desc_list.extend(parts)
                    
                    # Remove duplicates but keep order
                    unique_descs = []
                    for desc in desc_list:
                        if desc not in unique_descs:
                            unique_descs.append(desc)
                    
                    if unique_descs:  # Only include if we have descriptors
                        molecules.append({
                            'cid': int(cid),
                            'smiles': row.get('SMILES', ''),
                            'name': row.get('Name', f'CID_{cid}'),
                            'descriptors': unique_descs,
                            'source': 'goodscents'
                        })
    
    print(f"\nCreated {len(molecules)} molecule records")
    
    # Save to JSON for development
    output_path = Path('data/processed')
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'molecules.json', 'w') as f:
        json.dump(molecules, f, indent=2)
    
    print(f"Saved to {output_path / 'molecules.json'}")
    return molecules

def create_synthetic_dataset():
    """Create a synthetic dataset for testing if real data fails"""
    print("Creating synthetic dataset for development...")
    
    synthetic_molecules = [
        {
            'cid': 1001,
            'smiles': 'CCCC(=O)OC',
            'name': 'Ethyl Butyrate',
            'descriptors': ['fruity', 'sweet', 'pineapple'],
            'source': 'synthetic'
        },
        {
            'cid': 1002,
            'smiles': 'CC(C)CC=O',
            'name': 'Isobutyraldehyde',
            'descriptors': ['pungent', 'fruity', 'green'],
            'source': 'synthetic'
        },
        {
            'cid': 1003,
            'smiles': 'C1=CC=C(C=C1)C=O',
            'name': 'Benzaldehyde',
            'descriptors': ['almond', 'cherry', 'sweet'],
            'source': 'synthetic'
        },
        {
            'cid': 1004,
            'smiles': 'CC(=O)C',
            'name': 'Acetone',
            'descriptors': ['sweet', 'fruity', 'ethereal'],
            'source': 'synthetic'
        },
        {
            'cid': 1005,
            'smiles': 'CCCCC(=O)O',
            'name': 'Valeric Acid',
            'descriptors': ['pungent', 'cheesy', 'sweaty'],
            'source': 'synthetic'
        },
        {
            'cid': 1006,
            'smiles': 'CC1CCC(C(=C)C2C)CC2(C)C',
            'name': 'Pinene',
            'descriptors': ['pine', 'woody', 'fresh'],
            'source': 'synthetic'
        },
        {
            'cid': 1007,
            'smiles': 'CC(C)Cc1ccc(cc1)O',
            'name': 'Thymol',
            'descriptors': ['medicinal', 'herbal', 'spicy'],
            'source': 'synthetic'
        },
        {
            'cid': 1008,
            'smiles': 'CC(=CC(=O)C)C',
            'name': 'Mesityl Oxide',
            'descriptors': ['honey', 'sweet', 'caramel'],
            'source': 'synthetic'
        }
    ]
    
    # Save synthetic data
    output_path = Path('data/processed')
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'molecules.json', 'w') as f:
        json.dump(synthetic_molecules, f, indent=2)
    
    print(f"Created {len(synthetic_molecules)} synthetic molecules")
    return synthetic_molecules

if __name__ == "__main__":
    print("OpenSmell Data Setup")
    print("=" * 50)
    
    # Explore available data
    datasets = explore_pyrfume_data()
    
    # Create initial dataset
    molecules = create_initial_dataset()
    
    if molecules:
        print("\nSample molecules:")
        for i, mol in enumerate(molecules[:3]):
            print(f"\n{i+1}. {mol['name']} (CID: {mol['cid']})")
            print(f"   SMILES: {mol['smiles']}")
            print(f"   Descriptors: {', '.join(mol['descriptors'])}")