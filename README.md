
# OpenSmell Search Engine

A text-to-scent search engine that allows users to query a database of molecules using natural language odor descriptions.

## 🚀 Quick Start

### Local Development

1. **Clone and setup:**
```bash
git clone <repository-url>
cd scent_search
pip install -r requirements.txt
Initialize data:

bash
python setup_data.py
python build_vector_db.py
Start the API server:

bash
cd api
uvicorn main:app --reload
Start the Streamlit UI (in another terminal):

bash
streamlit run streamlit_app.py
Docker Deployment
bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individually
docker build -t opensmell .
docker run -p 8000:8000 -p 8501:8501 opensmell
🌐 Access Points
API Documentation: http://localhost:8000/docs

Streamlit UI: http://localhost:8501

API Base URL: http://localhost:8000

🔧 API Endpoints
Search
text
GET /search?q=fruity+and+sweet&top_k=10
Health Check
text
GET /health
Database Info
text
GET /info
Get Molecule by CID
text
GET /molecule/12345
Find Similar Molecules
text
GET /similar/12345?top_k=5
📊 Data Sources
Primary: Pyrfume Project (The Good Scents Company)

Format: Molecules with CID, SMILES, names, and odor descriptors

Processing: Descriptors are vectorized using Sentence Transformers

🧠 Search Methodology
Query Processing: User's natural language query is embedded using all-MiniLM-L6-v2

Scent Profiling: Each molecule's descriptors are averaged into a scent profile vector

Similarity Search: FAISS index performs efficient cosine similarity search

Ranking: Results sorted by similarity score (0-1)

🗂️ Project Structure
text
opensmell-search/
├── api/                    # FastAPI backend
│   ├── main.py            # API server
│   └── requirements-api.txt
├── data/                  # Data storage
│   ├── processed/        # Cleaned data
│   └── vector_db/       # FAISS indices & vectors
├── streamlit_app.py      # Web interface
├── setup_data.py         # Data initialization
├── build_vector_db.py    # Vector database builder
├── requirements.txt      # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
🔮 Future Roadmap
Phase 2: Enhanced Features
User accounts and saved searches

Molecule structure visualization

Community odor descriptions

Advanced filtering (molecular weight, volatility)

Phase 3: API for Makers
RESTful API for e-nose devices

Real-time sensor data processing

Multi-modal search (text + sensor data)

Commercial API tier

Phase 4: Advanced ML
Fine-tuned odor embedding model

Cross-modal retrieval (image/sound to scent)

Predictive odor generation

Large-scale odor mapping

📈 Monetization Path
Phase 1: Open source, community building

Phase 2: Premium features (advanced search, analytics)

Phase 3: API-as-a-Service for e-nose makers

Phase 4: Enterprise solutions (perfume, food industry)

🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines.

Fork the repository

Create a feature branch

Submit a Pull Request

📄 License
MIT License - see LICENSE file for details.

🙏 Acknowledgments
Pyrfume Project for odor data

Sentence Transformers for embedding models

FAISS for efficient similarity search

Streamlit & FastAPI communities

text

## **Quick Start Commands**

```bash
# 1. Clone and setup (if using version control)
git clone https://github.com/yourusername/opensmell-search.git
cd opensmell-search

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup data
python setup_data.py

# 5. Build vector database
python build_vector_db.py

# 6. Start API server (Terminal 1)
cd api
uvicorn main:app --reload

# 7. Start UI (Terminal 2)
cd ..
streamlit run streamlit_app.py
Testing the System
Once everything is running:

Test API:

Open browser to http://localhost:8000/docs for interactive API docs

Try: http://localhost:8000/search?q=fruity+and+sweet

Test UI:

Open browser to http://localhost:8501

Search for "fruity sweet", "woody smoky", etc.

Verify data:

Check http://localhost:8000/info for database stats

Inspect data/vector_db/ for generated files

Next Steps & Phase 2 Preparation
Enhance data pipeline:

Add more datasets from Pyrfume

Implement data versioning

Add data quality checks

Improve search quality:

Experiment with different embedding models

Implement query expansion

Add relevance feedback

Scale infrastructure:

Set up PostgreSQL for molecule metadata

Implement Redis caching for frequent queries

Add monitoring and logging

This implementation gives you a working MVP that can be deployed today. The architecture is modular, allowing you to easily swap components as you scale.

Would you like me to elaborate on any specific part or help you with deployment to a specific platform (Hugging Face Spaces, Streamlit Cloud, etc.)?