import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

class ScentVectorizer:
    """Convert scent descriptions to vectors and build search index"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with a sentence transformer model"""
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
    def vectorize_descriptors(self, descriptors: List[str]) -> np.ndarray:
        """Convert a list of descriptors to a single scent profile vector"""
        if not descriptors:
            return np.zeros(self.embedding_dim)
        
        # Embed each descriptor
        descriptor_embeddings = self.model.encode(
            descriptors,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Average embeddings to create scent profile
        scent_vector = np.mean(descriptor_embeddings, axis=0)
        
        # Normalize for cosine similarity
        scent_vector = normalize(scent_vector.reshape(1, -1))[0]
        
        return scent_vector
    
    def vectorize_query(self, query: str) -> np.ndarray:
        """Convert a natural language query to a vector"""
        # Enhanced query processing
        query_lower = query.lower().strip()
        
        # If query is simple, try to enrich it
        if len(query_lower.split()) < 3:
            # Common scent qualifiers to enhance simple queries
            scent_qualifiers = {
                'fruity': ['ripe', 'sweet', 'juicy', 'fresh'],
                'floral': ['blooming', 'fresh', 'sweet', 'delicate'],
                'woody': ['dry', 'earthy', 'warm', 'smoky'],
                'sweet': ['sugary', 'honeyed', 'candy-like', 'syrupy'],
                'citrus': ['fresh', 'zesty', 'tangy', 'bright']
            }
            
            # Add relevant qualifiers
            enhanced_parts = [query_lower]
            for key, qualifiers in scent_qualifiers.items():
                if key in query_lower:
                    enhanced_parts.extend(qualifiers[:2])
            
            query_lower = " ".join(set(enhanced_parts))
        
        # Vectorize the (possibly enhanced) query
        query_vector = self.model.encode(
            query_lower,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        query_vector = normalize(query_vector.reshape(1, -1))[0]
        
        return query_vector

class ScentDatabase:
    """Manage scent vector database and similarity search"""
    
    def __init__(self, data_path: Path = Path('data/processed')):
        self.data_path = data_path
        self.vectorizer = ScentVectorizer()
        self.molecules: List[Dict] = []
        self.vectors: np.ndarray = None
        self.index = None
        
    def load_molecules(self) -> List[Dict]:
        """Load molecule data from JSON"""
        json_path = self.data_path / 'molecules.json'
        
        if not json_path.exists():
            raise FileNotFoundError(f"Data file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            self.molecules = json.load(f)
        
        print(f"Loaded {len(self.molecules)} molecules")
        return self.molecules
    
    def build_vector_database(self) -> None:
        """Build vector representations for all molecules"""
        if not self.molecules:
            self.load_molecules()
        
        print("Building scent profile vectors...")
        
        # Create vectors for each molecule
        vectors = []
        valid_molecules = []
        
        for i, mol in enumerate(self.molecules):
            descriptors = mol.get('descriptors', [])
            if descriptors:
                # Vectorize the scent profile
                scent_vector = self.vectorizer.vectorize_descriptors(descriptors)
                vectors.append(scent_vector)
                valid_molecules.append(mol)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(self.molecules)} molecules")
        
        # Convert to numpy array
        self.vectors = np.array(vectors).astype('float32')
        self.molecules = valid_molecules
        
        print(f"Created {len(self.vectors)} scent vectors")
        print(f"Vector shape: {self.vectors.shape}")
    
    def build_faiss_index(self) -> None:
        """Build FAISS index for efficient similarity search"""
        if self.vectors is None:
            self.build_vector_database()
        
        print("Building FAISS index...")
        
        # Create index (using Inner Product for cosine similarity since vectors are normalized)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for similar scents"""
        if self.index is None:
            self.build_faiss_index()
        
        # Vectorize query
        query_vector = self.vectorizer.vectorize_query(query)
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, min(top_k, len(self.molecules)))
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                molecule = self.molecules[idx].copy()
                molecule['similarity_score'] = float(distance)
                molecule['rank'] = i + 1
                results.append(molecule)
        
        return results
    
    def save(self, output_dir: Path = Path('data/vector_db')) -> None:
        """Save the database to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save molecules and vectors
        with open(output_dir / 'molecules.pkl', 'wb') as f:
            pickle.dump(self.molecules, f)
        
        with open(output_dir / 'vectors.npy', 'wb') as f:
            np.save(f, self.vectors)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_dir / 'faiss_index.bin'))
        
        # Save metadata
        metadata = {
            'num_molecules': len(self.molecules),
            'vector_shape': self.vectors.shape,
            'embedding_dim': self.vectors.shape[1],
            'model_name': 'all-MiniLM-L6-v2'
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Database saved to {output_dir}")
    
    def load(self, input_dir: Path = Path('data/vector_db')) -> None:
        """Load database from disk"""
        if not input_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {input_dir}")
        
        # Load molecules and vectors
        with open(input_dir / 'molecules.pkl', 'rb') as f:
            self.molecules = pickle.load(f)
        
        with open(input_dir / 'vectors.npy', 'rb') as f:
            self.vectors = np.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(str(input_dir / 'faiss_index.bin'))
        
        print(f"Loaded database with {len(self.molecules)} molecules")

def main():
    """Build and save the scent vector database"""
    print("OpenSmell Scent Vector Database Builder")
    print("=" * 50)
    
    # Initialize database
    db = ScentDatabase()
    
    # Build vector database
    db.build_vector_database()
    
    # Build FAISS index
    db.build_faiss_index()
    
    # Test search
    test_queries = [
        "fruity and sweet",
        "woody and smoky",
        "fresh citrus",
        "floral and romantic"
    ]
    
    print("\nTest Searches:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = db.search(query, top_k=3)
        for result in results:
            print(f"  - {result['name']}: {result['descriptors']} (score: {result['similarity_score']:.3f})")
    
    # Save database
    db.save()
    
    print("\nDatabase build complete!")

if __name__ == "__main__":
    main()