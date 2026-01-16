import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Scent Search Engine",
    page_icon="👃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4B0082;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .similarity-badge {
        background-color: #4B0082;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .descriptor-tag {
        display: inline-block;
        background-color: #e0e0e0;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"  # Change this for production

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def search_molecules(query: str, top_k: int = 10):
    """Search for molecules using the API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/search",
            params={"q": query, "top_k": top_k, "include_metadata": True}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_database_info():
    """Get database information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def display_molecule_card(molecule: Dict, rank: int):
    """Display a molecule result card"""
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 2])
        
        with col1:
            st.markdown(f"**{rank}. {molecule['name']}**")
            st.caption(f"CID: {molecule['cid']}")
            
            # Display descriptors as tags
            if 'descriptors' in molecule:
                descriptor_html = ""
                for desc in molecule['descriptors'][:6]:  # Show first 6 descriptors
                    descriptor_html += f'<span class="descriptor-tag">{desc}</span> '
                st.markdown(descriptor_html, unsafe_allow_html=True)
        
        with col2:
            similarity = molecule.get('similarity_score', 0)
            st.markdown(f'<div class="similarity-badge">{similarity:.3f}</div>', 
                       unsafe_allow_html=True)
            st.caption("Similarity")
        
        with col3:
            if 'smiles' in molecule:
                st.code(molecule['smiles'], language="text")
                st.caption("SMILES")
            
            if molecule.get('source'):
                st.caption(f"Source: {molecule['source']}")
        
        st.divider()

def create_similarity_plot(results: List[Dict]):
    """Create a bar chart of similarity scores"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    df = df.head(10)  # Show top 10
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['name'],
        y=df['similarity_score'],
        text=df['similarity_score'].round(3),
        textposition='auto',
        marker_color='#4B0082',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Top Results Similarity Scores",
        xaxis_title="Molecule",
        yaxis_title="Similarity Score",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_descriptor_network(results: List[Dict]):
    """Create a visualization of descriptor relationships"""
    if not results:
        return None
    
    # Collect all descriptors from top results
    all_descriptors = {}
    for result in results[:5]:
        for desc in result.get('descriptors', [])[:5]:  # Top 5 descriptors per molecule
            all_descriptors[desc] = all_descriptors.get(desc, 0) + 1
    
    if not all_descriptors:
        return None
    
    # Prepare data for treemap
    labels = list(all_descriptors.keys())
    parents = [""] * len(labels)
    values = list(all_descriptors.values())
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        textinfo="label+value",
        marker=dict(
            colors=values,
            colorscale='Purples',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title="Descriptor Frequency in Results",
        height=400,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">👃 OpenSmell Search Engine</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Search for molecules using natural language odor descriptions</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/perfume-bottle.png", 
                 width=80)
        
        st.markdown("### About")
        st.info("""
        This is a scent search engine that finds 
        molecules based on their odor descriptions.
        
        **Examples:**
        - "fruity and sweet"
        - "woody and smoky"
        - "fresh citrus aroma"
        - "floral and romantic"
        """)
        
        # Database info
        if check_api_health():
            info = get_database_info()
            if info:
                st.success(f"✅ Database loaded: {info.get('num_molecules', 0)} molecules")
        else:
            st.error("⚠️ API not connected")
            st.info("Start the API server with: `uvicorn api.main:app --reload`")
        
        st.divider()
        
        # Advanced options
        st.markdown("### Settings")
        top_k = st.slider("Number of results", 5, 50, 10)
        min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.0, 0.1)
        
        st.divider()
        
        # Quick examples
        st.markdown("### Quick Examples")
        example_queries = [
            "fruity sweet",
            "woody smoky",
            "citrus fresh",
            "floral romantic",
            "herbal medicinal"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query}"):
                st.session_state.search_query = query
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search input
        search_query = st.text_input(
            "🔍 Describe an odor:",
            value=st.session_state.get("search_query", ""),
            placeholder="e.g., 'fruity and sweet' or 'smoky like a campfire'",
            key="search_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Perform search
    if search_button and search_query:
        with st.spinner("Searching for matching scents..."):
            results = search_molecules(search_query, top_k=top_k)
            
            if results:
                # Display search results
                st.markdown(f"### Found {results['num_results']} results for: '{results['query']}'")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Results", "Visualizations", "Raw Data"])
                
                with tab1:
                    # Filter by minimum score
                    filtered_results = [
                        r for r in results['results'] 
                        if r['similarity_score'] >= min_score
                    ]
                    
                    if not filtered_results:
                        st.warning(f"No results meet the minimum score of {min_score}")
                    else:
                        for i, result in enumerate(filtered_results):
                            display_molecule_card(result, i + 1)
                
                with tab2:
                    if results['results']:
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            fig1 = create_similarity_plot(results['results'])
                            if fig1:
                                st.plotly_chart(fig1, use_container_width=True)
                        
                        with col_viz2:
                            fig2 = create_descriptor_network(results['results'])
                            if fig2:
                                st.plotly_chart(fig2, use_container_width=True)
                        
                        # Additional statistics
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        
                        with col_stats1:
                            avg_score = np.mean([r['similarity_score'] for r in results['results']])
                            st.metric("Average Similarity", f"{avg_score:.3f}")
                        
                        with col_stats2:
                            top_score = results['results'][0]['similarity_score']
                            st.metric("Top Score", f"{top_score:.3f}")
                        
                        with col_stats3:
                            unique_desc = set()
                            for r in results['results']:
                                unique_desc.update(r['descriptors'])
                            st.metric("Unique Descriptors", len(unique_desc))
                
                with tab3:
                    st.json(results)
            else:
                st.error("No results found or API connection failed")
    
    elif not search_query and search_button:
        st.warning("Please enter a search query")
    
    # Display initial state or instructions
    if not search_button or not search_query:
        st.markdown("---")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("### 🎯 How to Search")
            st.markdown("""
            1. Describe an odor in natural language
            2. Use adjectives like fruity, sweet, smoky
            3. Combine multiple descriptors
            4. Click Search or press Enter
            """)
        
        with col_info2:
            st.markdown("### 📊 Understanding Results")
            st.markdown("""
            **Similarity Score:** 0-1, higher is better match
            
            **Descriptors:** Odor tags for each molecule
            
            **CID:** PubChem Compound ID
            
            **SMILES:** Chemical structure notation
            """)
        
        with col_info3:
            st.markdown("### 🔬 Data Sources")
            st.markdown("""
            - The Good Scents Company
            - Pyrfume Project datasets
            - Scientific odor databases
            
            Currently tracking **1000+** odor molecules
            """)
        
        # Quick start examples
        st.markdown("### 💡 Try These Examples")
        example_cols = st.columns(5)
        examples = [
            ("🍓", "berry fruity"),
            ("🌹", "rose floral"),
            ("🪵", "woody earthy"),
            ("🍋", "citrus zesty"),
            ("☕", "coffee roasted")
        ]
        
        for i, (icon, query) in enumerate(examples):
            with example_cols[i]:
                if st.button(f"{icon}\n{query}", use_container_width=True):
                    st.session_state.search_query = query
                    st.rerun()

if __name__ == "__main__":
    # Initialize session state
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    
    main()