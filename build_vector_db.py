"""
OpenSmell - Build Vector Database from Pyrfume Data
This version processes the real, standardized datasets you downloaded.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import faiss
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import time

class ScentDatabase:
    """Build and manage a unified scent database from multiple Pyrfume datasets"""
    
    def __init__(self, pyrfume_cache_dir: str = None):
        """
        Initialize the database builder.
        
        Args:
            pyrfume_cache_dir: Path to Pyrfume's cache directory. 
                               Defaults to ~/.pyrfume/data/
        """
        self.cache_dir = Path(pyrfume_cache_dir) if pyrfume_cache_dir else Path.home() / '.pyrfume' / 'data'
        self.output_dir = Path("data/full_database")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
        
        # Data storage
        self.molecules = []  # List of dicts with molecule info
        self.cid_to_index = {}  # Map PubChem CID to index in self.molecules
        self.descriptor_texts = []  # Text descriptions for embedding
        self.fingerprints = []  # Morgan fingerprints
        
    def find_and_load_datasets(self, dataset_limit=None):
        """Discover and load available Pyrfume datasets."""
        try:
            import pyrfume
            all_archive_names = pyrfume.list_archives()
            print(f"Found {len(all_archive_names)} total dataset archives.")
            
            # Filter out known non-dataset or problematic entries
            known_issues = ['embedding', 'knapsack', 'mordred', 'morgan', 'nhanes_2014']
            dataset_names = [name for name in all_archive_names if name not in known_issues]
            
            if dataset_limit:
                dataset_names = dataset_names[:dataset_limit]
                print(f"Processing first {dataset_limit} datasets...")
            
            total_added = 0
            for dataset in dataset_names:
                added = self.load_dataset(dataset)
                total_added += added
            
            print(f"Total: Added {total_added} unique molecules from {len(dataset_names)} datasets")
            return total_added
            
        except Exception as e:
            print(f"Error discovering datasets: {e}")
            return 0
    
    def load_dataset(self, dataset_name: str):
        """Load and process a single Pyrfume dataset using the library's data loader."""
        try:
            # Use pyrfume.load_data() to fetch and cache the dataset files
            import pyrfume
            molecules_df = pyrfume.load_data(f'{dataset_name}/molecules.csv')
            print(f"Loaded {dataset_name}: {len(molecules_df)} molecules")

            # Try to load behavior data; some datasets use stimuli.csv instead
            behavior_df = None
            try:
                behavior_df = pyrfume.load_data(f'{dataset_name}/behavior.csv')
                print(f"  Found behavior data with shape: {behavior_df.shape}")
            except Exception:
                # Fallback to stimuli.csv if behavior.csv doesn't exist
                try:
                    behavior_df = pyrfume.load_data(f'{dataset_name}/stimuli.csv')
                    print(f"  Found stimuli data with shape: {behavior_df.shape}")
                except Exception as e:
                    print(f"  Note: No behavior or stimuli file for {dataset_name} ({e})")
                    # Continue without behavior data - we'll just add molecules

            added_count = 0
            
            # Process each molecule in the dataset
            for _, mol_row in molecules_df.iterrows():
                cid = mol_row.get('CID')
                if pd.isna(cid):
                    continue
                    
                cid = int(cid)
                
                # If we haven't seen this CID before, add it to our database
                if cid not in self.cid_to_index:
                    # Create molecule entry
                    molecule = {
                        'cid': cid,
                        'name': mol_row.get('Name', f'CID_{cid}'),
                        'smiles': mol_row.get('SMILES', ''),
                        'descriptors': [],
                        'sources': [dataset_name],
                        'descriptor_details': {}
                    }
                    
                    # Extract descriptors from behavior data if available
                    if behavior_df is not None:
                        # Determine how to find this molecule in the behavior data
                        # Different datasets use different column names
                        id_column = None
                        for possible_id in ['Stimulus', 'CID', 'StimulusID', 'cid']:
                            if possible_id in behavior_df.columns:
                                id_column = possible_id
                                break
                        
                        if id_column:
                            # Find behavior data for this specific molecule
                            mol_behavior = behavior_df[behavior_df[id_column] == cid]
                            
                            if not mol_behavior.empty:
                                # Identify descriptor columns (exclude ID and metadata columns)
                                exclude_cols = ['Stimulus', 'CID', 'Index', 'StimulusID', 'cid', 
                                              'Name', 'Concentration', 'Units', 'Replicate']
                                descriptor_cols = [col for col in behavior_df.columns 
                                                 if col not in exclude_cols]
                                
                                # Add descriptors with positive scores
                                for desc in descriptor_cols:
                                    if desc in mol_behavior.columns:
                                        score = mol_behavior[desc].iloc[0]
                                        if pd.notna(score):
                                            try:
                                                score_val = float(score)
                                                if score_val > 0:
                                                    molecule['descriptors'].append(desc.lower())
                                                    key = f"{dataset_name}:{desc}"
                                                    molecule['descriptor_details'][key] = score_val
                                            except (ValueError, TypeError):
                                                # Handle non-numeric scores
                                                if str(score).strip().lower() not in ['', 'na', 'nan', 'none']:
                                                    molecule['descriptors'].append(desc.lower())
                                                    key = f"{dataset_name}:{desc}"
                                                    molecule['descriptor_details'][key] = str(score)
                    
                    # Add the molecule to our database
                    self.molecules.append(molecule)
                    self.cid_to_index[cid] = len(self.molecules) - 1
                    added_count += 1
                else:
                    # Update existing molecule with new source
                    idx = self.cid_to_index[cid]
                    if dataset_name not in self.molecules[idx]['sources']:
                        self.molecules[idx]['sources'].append(dataset_name)
            
            print(f"  Added {added_count} new molecules to database")
            return added_count
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def build_descriptor_texts(self):
        """Create text descriptions for each molecule for embedding"""
        print("\nCreating text descriptions for embedding...")
        
        for molecule in tqdm(self.molecules, desc="Processing molecules"):
            # Create a rich text description from all available data
            parts = []
            
            # Add descriptors
            if molecule['descriptors']:
                # Take top descriptors by frequency or score
                top_descriptors = list(dict.fromkeys(molecule['descriptors']))[:10]
                parts.append(f"Odor descriptors: {', '.join(top_descriptors)}")
            
            # Add source information
            if molecule['sources']:
                parts.append(f"Data sources: {', '.join(molecule['sources'][:3])}")
            
            # Create final text
            text = ". ".join(parts) if parts else "No odor description available"
            self.descriptor_texts.append(text)
            
            # Also store the text in the molecule dict for display
            molecule['description_text'] = text
    
    def generate_morgan_fingerprints(self):
        """Generate Morgan fingerprints for all molecules with SMILES"""
        print("\nGenerating Morgan fingerprints...")
        
        valid_count = 0
        for molecule in tqdm(self.molecules, desc="Generating fingerprints"):
            smiles = molecule.get('smiles', '')
            
            if pd.isna(smiles) or not str(smiles).strip():
                self.fingerprints.append(None)
                continue
                
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fp_array = np.zeros((0,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, fp_array)
                    self.fingerprints.append(fp_array.astype('float32'))
                    valid_count += 1
                else:
                    self.fingerprints.append(None)
            except:
                self.fingerprints.append(None)
        
        print(f"Generated fingerprints for {valid_count}/{len(self.molecules)} molecules")
    
    def build_vector_embeddings(self):
        """Create sentence embeddings from descriptor texts"""
        print("\nCreating sentence embeddings...")
        
        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(self.descriptor_texts):
            if text and text != "No odor description available":
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            print("Warning: No valid texts for embedding")
            return None
        
        # Create embeddings in batches
        batch_size = 32
        all_embeddings = np.zeros((len(self.descriptor_texts), self.model.get_sentence_embedding_dimension()))
        
        for i in tqdm(range(0, len(valid_texts), batch_size), desc="Encoding batches"):
            batch = valid_texts[i:i+batch_size]
            batch_indices = valid_indices[i:i+batch_size]
            
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            
            for j, idx in enumerate(batch_indices):
                all_embeddings[idx] = embeddings[j]
        
        return all_embeddings.astype('float32')
    
    def build_faiss_indexes(self, embeddings, valid_fingerprints, valid_fp_indices):
        """Build FAISS indexes for both embeddings and fingerprints"""
        print("\nBuilding FAISS indexes...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Descriptor embedding index
        if embeddings is not None and len(embeddings) > 0:
            desc_index = faiss.IndexFlatIP(embeddings.shape[1])
            desc_index.add(embeddings)
            faiss.write_index(desc_index, str(self.output_dir / "descriptor_index.faiss"))
            print(f"Descriptor index: {desc_index.ntotal} vectors")
        
        # 2. Fingerprint index
        if valid_fingerprints and len(valid_fingerprints) > 0:
            fp_matrix = np.vstack(valid_fingerprints)
            fp_index = faiss.IndexFlatIP(fp_matrix.shape[1])
            fp_index.add(fp_matrix)
            faiss.write_index(fp_index, str(self.output_dir / "fingerprint_index.faiss"))
            
            # Save mapping from fingerprint index to molecule index
            fp_to_mol = {i: valid_fp_indices[i] for i in range(len(valid_fp_indices))}
            with open(self.output_dir / "fp_mapping.json", 'w') as f:
                json.dump(fp_to_mol, f)
            
            print(f"Fingerprint index: {fp_index.ntotal} vectors")
    
    def save_database(self):
        """Save the complete database"""
        print("\nSaving database...")
        
        # Save molecules metadata
        molecules_light = []
        for mol in self.molecules:
            mol_light = {
                'cid': mol['cid'],
                'name': mol['name'],
                'smiles': mol.get('smiles', ''),
                'descriptors': mol['descriptors'],
                'sources': mol['sources'],
                'description_text': mol.get('description_text', '')
            }
            molecules_light.append(mol_light)
        
        with open(self.output_dir / "molecules.json", 'w') as f:
            json.dump(molecules_light, f, indent=2)
        
        # Save metadata
        metadata = {
            "num_molecules": len(self.molecules),
            "num_datasets": len(set([src for mol in self.molecules for src in mol['sources']])),
            "num_with_fingerprints": sum(1 for fp in self.fingerprints if fp is not None),
            "build_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "all-MiniLM-L6-v2",
            "fingerprint_type": "Morgan_2048bit_radius2"
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Database saved to {self.output_dir}")
        print(f"Contains {len(self.molecules)} unique molecules")
    
    def build(self, dataset_limit: int = None):
        """Main build process"""
        print("="*70)
        print("BUILDING OPEN SMELL DATABASE FROM PYRFUME DATA")
        print("="*70)
        
        total_start = time.time()
        
        # Step 1: Find and load datasets
        total_added = self.find_and_load_datasets(dataset_limit)
        
        if total_added == 0:
            print("Warning: No molecules were loaded. Exiting build process.")
            return
        
        num_datasets = len(set([src for mol in self.molecules for src in mol['sources']]))
        print(f"Loaded {total_added} molecules from {num_datasets} datasets")
        
        # Step 2: Process descriptors
        self.build_descriptor_texts()
        
        # Step 3: Generate fingerprints
        self.generate_morgan_fingerprints()
        
        # Step 4: Create embeddings
        embeddings = self.build_vector_embeddings()
        
        # Step 5: Prepare fingerprint data for indexing
        valid_fingerprints = []
        valid_fp_indices = []
        for i, fp in enumerate(self.fingerprints):
            if fp is not None:
                valid_fingerprints.append(fp)
                valid_fp_indices.append(i)
        
        # Step 6: Build indexes
        self.build_faiss_indexes(embeddings, valid_fingerprints, valid_fp_indices)
        
        # Step 7: Save everything
        self.save_database()
        
        total_elapsed = time.time() - total_start
        print(f"Total build time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        
        # Display summary
        print("\n" + "="*70)
        print("DATABASE BUILD COMPLETE!")
        print("="*70)
        print(f"\nFINAL SUMMARY:")
        print(f"   Total unique molecules: {len(self.molecules)}")
        print(f"   Molecules with fingerprints: {sum(1 for fp in self.fingerprints if fp is not None)}")
        print(f"   Datasets processed: {num_datasets}")
        print(f"   Database location: {self.output_dir}")
        
        # Show sample molecules
        print(f"\nSAMPLE MOLECULES:")
        for i, mol in enumerate(self.molecules[:3]):
            print(f"   {i+1}. {mol['name']} (CID: {mol['cid']})")
            print(f"      SMILES: {mol.get('smiles', 'N/A')}")
            print(f"      Descriptors: {', '.join(mol['descriptors'][:3])}" + 
                  ("..." if len(mol['descriptors']) > 3 else ""))
            print(f"      Sources: {', '.join(mol['sources'][:2])}" + 
                  ("..." if len(mol['sources']) > 2 else ""))
            print()

def main():
    """Main function to build the database"""
    builder = ScentDatabase()
    
    # Optional: Limit to first N datasets for testing
    # builder.build(dataset_limit=2)
    
    # To build with all datasets (uncomment when ready):
    builder.build()

if __name__ == "__main__":
    main()