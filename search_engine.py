"""
PHASE 3: IMMEDIATE UTILITY - ODOR SEARCH ENGINE
A simple web API to query your odor index.
Run: uvicorn phase3_search_engine:app --reload
Then visit: http://localhost:8000
"""
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import pandas as pd
from typing import List, Optional

app = FastAPI(title="OpenSmell Search", description="Search chemicals by odor and odors by chemical")

# Load your index
INDEX_PATH = Path("data/pyrfume_structured/odor_search_index.json")
with open(INDEX_PATH, 'r') as f:
    ODOR_INDEX = json.load(f)

@app.get("/", response_class=HTMLResponse)
def home():
    """Simple search interface"""
    return """
    <html>
        <head><title>OpenSmell Search</title></head>
        <body style="font-family: sans-serif; padding: 2rem;">
            <h1>🧪 OpenSmell Odor Search</h1>
            <p>Search <strong>{total_molecules}</strong> chemicals with <strong>{total_descriptors}</strong> unique odor descriptors.</p>
            
            <h3>1. Find Chemicals by Odor</h3>
            <form action="/search/odor">
                <input type="text" name="q" placeholder="e.g., citrus, fruity, floral" style="width: 300px; padding: 0.5rem;">
                <button type="submit">Search</button>
            </form>
            
            <h3>2. Find Odors by Chemical</h3>
            <form action="/search/chemical">
                <input type="text" name="q" placeholder="e.g., limonene, vanillin, CID_179" style="width: 300px; padding: 0.5rem;">
                <button type="submit">Search</button>
            </form>
            
            <h3>3. Export Data</h3>
            <p><a href="/api/export/csv">Download full dataset as CSV</a></p>
            <p><a href="/api/export/json">Download full dataset as JSON</a></p>
            
            <hr>
            <p><em>Data from Pyrfume archives: {sources_count} sources. Built with OpenSmell.</em></p>
        </body>
    </html>
    """.format(
        total_molecules=len(ODOR_INDEX),
        total_descriptors=len(set(d for m in ODOR_INDEX for d in m['descriptors'])),
        sources_count=len(set(s for m in ODOR_INDEX for s in m['sources']))
    )

@app.get("/search/odor")
def search_by_odor(q: str = Query(..., description="Odor descriptor(s), comma-separated")):
    """Find chemicals containing specific odor descriptors"""
    query_terms = [term.strip().lower() for term in q.split(',')]
    
    results = []
    for chemical in ODOR_INDEX:
        matches = []
        for term in query_terms:
            # Check if term appears in any descriptor (partial match)
            for desc in chemical['descriptors']:
                if term in desc.lower():
                    matches.append(desc)
                    break
        
        if len(matches) == len(query_terms):  # All terms matched
            results.append({
                "cid": chemical['cid'],
                "name": chemical['name'],
                "smiles": chemical['smiles'],
                "matched_descriptors": matches,
                "all_descriptors": chemical['descriptors'][:10],  # First 10
                "source_count": len(chemical['sources'])
            })
    
    return {
        "query": q,
        "results_count": len(results),
        "results": results[:50]  # Limit for demo
    }

@app.get("/search/chemical")
def search_by_chemical(q: str = Query(..., description="Chemical name, CID, or SMILES substring")):
    """Find odors for a specific chemical"""
    q_lower = q.lower()
    results = []
    
    for chemical in ODOR_INDEX:
        # Search in CID, name, and SMILES
        if (str(chemical['cid']) == q or 
            q_lower in chemical['name'].lower() or 
            q_lower in chemical['smiles'].lower()):
            
            results.append({
                "cid": chemical['cid'],
                "name": chemical['name'],
                "smiles": chemical['smiles'],
                "descriptors": chemical['descriptors'],
                "sources": chemical['sources'],
                "descriptor_count": len(chemical['descriptors'])
            })
    
    return {
        "query": q,
        "results_count": len(results),
        "results": results
    }

@app.get("/api/export/csv")
def export_csv():
    """Export the full index as CSV"""
    df_data = []
    for chem in ODOR_INDEX:
        df_data.append({
            "cid": chem['cid'],
            "name": chem['name'],
            "smiles": chem['smiles'],
            "descriptors": " | ".join(chem['descriptors']),
            "descriptor_count": len(chem['descriptors']),
            "sources": ", ".join(chem['sources'])
        })
    
    df = pd.DataFrame(df_data)
    csv_content = df.to_csv(index=False)
    return JSONResponse(content={"csv": csv_content})

@app.get("/api/stats")
def get_stats():
    """Get dataset statistics"""
    all_descriptors = [d for chem in ODOR_INDEX for d in chem['descriptors']]
    unique_descriptors = set(all_descriptors)
    
    # Most common descriptors
    from collections import Counter
    desc_counts = Counter(all_descriptors)
    
    return {
        "total_chemicals": len(ODOR_INDEX),
        "chemicals_with_descriptors": sum(1 for c in ODOR_INDEX if c['descriptors']),
        "total_descriptor_assignments": len(all_descriptors),
        "unique_descriptors": len(unique_descriptors),
        "top_descriptors": desc_counts.most_common(20),
        "archives_represented": len(set(s for c in ODOR_INDEX for s in c['sources']))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)