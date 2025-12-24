# OpenSmell Scent Search Engine

The Search Engine for Scents, Smells and Olfactory Data

## Overview

OpenSmell is a proof-of-concept application that bridges natural language and chemical informatics. It transforms textual odor descriptions into mathematical vectors and performs similarity searches against a database of molecular scent profiles. The system consists of a FastAPI backend for serving search queries and a Streamlit-based web interface.

## Features

*   **Natural Language Search**: Input descriptive odor queries to find relevant molecules.
*   **Vector-Based Similarity**: Uses sentence transformer models (`all-MiniLM-L6-v2`) to encode text and scent descriptors into a shared vector space.
*   **Efficient Retrieval**: Employs a FAISS index for fast nearest-neighbor search.
*   **Modern Web Stack**: Features a FastAPI backend with auto-generated documentation and an interactive Streamlit frontend.
*   **Modular Data Pipeline**: Designed for integration with olfactory datasets like the Pyrfume Project.

## Project Structure

```
opensmell-search/
├── api/                    # FastAPI backend application
│   ├── main.py            # Core API server and endpoints
│   └── requirements-api.txt
├── data/                  # Generated data storage
│   ├── processed/        # Cleaned and structured molecule data
│   └── vector_db/        # FAISS indices and embedding vectors
├── streamlit_app.py      # Primary web interface
├── setup_data.py         # Script to load and preprocess odorant data
├── build_vector_db.py    # Script to create the vector search database
├── requirements.txt      # Main Python dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Getting Started

### Prerequisites

*   Python 3.11 or 3.12
*   `pip` package manager

### Local Installation & Setup

1.  **Clone the repository and set up the environment:**
    ```bash
    git clone https://github.com/praisejamesx/scent_search.git
    cd scent_search
    python -m venv .venv
    # On Windows: .venv\Scripts\activate
    # On macOS/Linux: source .venv/bin/activate
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Initialize the data and build the search index:**
    ```bash
    python setup_data.py
    python build_vector_db.py
    ```
    *Note: The initial setup uses a synthetic dataset. See the Data Sources section for integrating real-world data.*

### Running the Application

1.  **Start the Backend API Server** (in one terminal):
    ```bash
    cd api
    uvicorn main:app --reload
    ```
    The API will be available at `http://localhost:8000`. Interactive documentation (Swagger UI) is at `http://localhost:8000/docs`.

2.  **Start the Web Interface** (in a separate terminal from the project root):
    ```bash
    streamlit run streamlit_app.py
    ```
    The Streamlit application will open in your browser at `http://localhost:8501`.

## Usage

### Using the Web Interface
Navigate to `http://localhost:8501`. Enter a natural language odor description (e.g., "fruity and sweet") into the search bar to retrieve a list of matching molecules ranked by similarity score.

### Core API Endpoints

| Endpoint | Method | Description | Parameters |
| :--- | :--- | :--- | :--- |
| `/search` | `GET` | Main search endpoint. | `q`: Query string.<br>`top_k`: Number of results (default: 10). |
| `/molecule/{cid}` | `GET` | Fetch details for a specific PubChem CID. | `cid`: PubChem Compound ID. |
| `/similar/{cid}` | `GET` | Find molecules similar to a given CID. | `cid`: Query CID.<br>`top_k`: Number of results. |
| `/health` | `GET` | Service health check. | |
| `/info` | `GET` | Basic database metadata. | |

## Technical Details

### Search Methodology
1.  **Query Embedding**: User input is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer.
2.  **Scent Profile Representation**: Each molecule's odor descriptors are individually embedded and averaged to form a single scent profile vector.
3.  **Similarity Matching**: The query vector is compared against all scent profile vectors using cosine similarity within a pre-built FAISS index.
4.  **Ranking**: Molecules are ranked by their similarity score (0 to 1, where 1 is most similar).

### Data Sources
The primary data source is the [Pyrfume Project](https://github.com/pyrfume/pyrfume-data), which provides structured databases linking molecules to odor descriptors (e.g., The Good Scents Company data). The current prototype includes a synthetic dataset for demonstration and development.

## Deployment

### Docker
A containerized setup is provided for easy deployment.

```bash
# Build and run using Docker Compose
docker-compose up --build

# Or build and run individually
docker build -t opensmell .
docker run -p 8000:8000 -p 8501:8501 opensmell
```

## Development Roadmap

*   **Data Integration**: Reliable ingestion of the full Pyrfume datasets and other public olfactory databases.
*   **Enhanced Features**: Advanced filtering (by molecular weight, volatility), molecular structure visualization, and user session management.
*   **Model Improvement**: Experimentation with specialized embedding models and fine-tuning for the olfactory domain.
*   **System Scalability**: Implementation of a dedicated database for metadata and caching layers for improved performance.
*   **E-Nose Integration**: Hardware integration to allow search from digital olfaction devices.

## Contributing
Contributions are welcome. Please feel free to fork the repository, create a feature branch, and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
*   [Pyrfume Project](https://pyrfume.org/) for curating and providing open access to olfactory data.
*   The developers of [Sentence Transformers](https://www.sbert.net/), [FAISS](https://github.com/facebookresearch/faiss), [FastAPI](https://fastapi.tiangolo.com/), and [Streamlit](https://streamlit.io/).