from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from build_vector_db import ScentDatabase

app = FastAPI(
    title="OpenSmell Search Engine API",
    description="Text-to-scent search engine for odor molecules",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = None

@app.on_event("startup")
async def startup_event():
    """Initialize the scent database on startup"""
    global db
    try:
        db = ScentDatabase()
        db.load()  # Try to load existing database
        print("Database loaded successfully")
    except Exception as e:
        print(f"Error loading database: {e}")
        print("Building new database...")
        db = ScentDatabase()
        db.build_vector_database()
        db.build_faiss_index()
        db.save()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenSmell Search Engine API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search?q=your_query",
            "health": "/health",
            "info": "/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "database_loaded": db is not None}

@app.get("/info")
async def get_database_info():
    """Get database information"""
    if db is None or db.molecules is None:
        raise HTTPException(status_code=503, detail="Database not loaded")
    
    return {
        "num_molecules": len(db.molecules),
        "sample_molecules": db.molecules[:5] if len(db.molecules) > 5 else db.molecules
    }

@app.get("/search")
async def search_molecules(
    q: str = Query(..., description="Search query (e.g., 'fruity and sweet')"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
    include_metadata: bool = Query(True, description="Include molecule metadata")
):
    """Search for molecules by odor description"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Perform search
        results = db.search(q, top_k=top_k)
        
        # Filter by minimum score
        if min_score > 0:
            results = [r for r in results if r['similarity_score'] >= min_score]
        
        # Format response
        response = {
            "query": q,
            "num_results": len(results),
            "results": []
        }
        
        for result in results:
            result_data = {
                "rank": result['rank'],
                "similarity_score": round(result['similarity_score'], 4),
                "name": result['name'],
                "cid": result['cid'],
                "descriptors": result['descriptors']
            }
            
            if include_metadata:
                result_data.update({
                    "smiles": result.get('smiles', ''),
                    "source": result.get('source', 'unknown'),
                    "descriptor_count": len(result['descriptors'])
                })
            
            response["results"].append(result_data)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/molecule/{cid}")
async def get_molecule_by_cid(cid: int):
    """Get molecule details by CID"""
    if db is None or db.molecules is None:
        raise HTTPException(status_code=503, detail="Database not loaded")
    
    for molecule in db.molecules:
        if molecule.get('cid') == cid:
            return molecule
    
    raise HTTPException(status_code=404, detail=f"Molecule with CID {cid} not found")

@app.get("/similar/{cid}")
async def find_similar_molecules(
    cid: int,
    top_k: int = Query(10, ge=1, le=100, description="Number of similar molecules to return")
):
    """Find molecules similar to a given molecule"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Find the target molecule
    target_idx = -1
    target_vector = None
    
    for idx, molecule in enumerate(db.molecules):
        if molecule.get('cid') == cid:
            target_idx = idx
            target_vector = db.vectors[idx].reshape(1, -1).astype('float32')
            break
    
    if target_idx == -1:
        raise HTTPException(status_code=404, detail=f"Molecule with CID {cid} not found")
    
    try:
        # Search for similar vectors
        distances, indices = db.index.search(target_vector, min(top_k + 1, len(db.molecules)))
        
        # Prepare results (skip the first one as it's the query molecule itself)
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != target_idx and idx != -1:  # Skip query molecule and invalid indices
                molecule = db.molecules[idx].copy()
                molecule['similarity_score'] = float(distance)
                molecule['rank'] = len(results) + 1
                results.append(molecule)
                if len(results) >= top_k:
                    break
        
        return {
            "query_cid": cid,
            "query_name": db.molecules[target_idx]['name'],
            "num_results": len(results),
            "results": results[:top_k]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar molecules: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)