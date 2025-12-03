# Research Analysis: Material Impact & Energy Absorption

A data analysis and machine learning pipeline for classifying and analyzing research papers related to material impact behavior, energy absorption, and structural analysis using embeddings and clustering techniques.

## Overview

This project processes and classifies research data across multiple domains including:
- **Concrete Strength Prediction** using machine learning
- **Sandwich Panels & Impact Analysis**
- **Composite Materials & Impact**
- **Crashworthiness & Energy Absorption**
- **Honeycomb Structures & Energy Absorption**
- **Reinforced Concrete & Impact**

## Project Structure

```
Research Analysis/
├── data_processing.ipynb          # Data loading, cleaning, and embedding generation
├── data_classification.ipynb      # Clustering, classification, and analysis
├── Data/                          # Input data and processed files
│   ├── Database_*.csv             # Raw research databases
│   └── data_with_embeddings.pkl   # Processed data with embeddings
├── Media/                         # Output visualizations and media
└── Final Database.xlsx            # Classified data exported by category
```

## Datasets

The project includes multiple research databases:

| Dataset | Description |
|---------|-------------|
| `Database_Concrete_AI_Impact.xlsx.csv` | Concrete materials with AI-based impact analysis |
| `Database_Concrete_Impact.xlsx.csv` | Concrete impact behavior studies |
| `Database_FEM_Impact_EnergyAbsorption.xlsx.csv` | Finite element method impact and energy absorption |
| `Database_Honeycomb_AI_Impact.xlsx.csv` | Honeycomb structures with AI analysis |
| `Database_Honeycomb_Concrete_Impact.xlsx.csv` | Honeycomb-concrete composite impact |
| `Database_Honeycomb_FEM_Impact.xlsx.csv` | Honeycomb FEM impact analysis |
| `Database_Honeycomb_Optimization_EnergyAbsorption.xlsx.csv` | Optimized honeycomb energy absorption |
| `Database_honeycomb structures_impact behavior_energy absorption.csv` | General honeycomb impact behavior |

## Pipeline Workflow

### 1. Data Processing (`data_processing.ipynb`)

- **Load & Consolidate**: Reads all CSV files from the `Data/` directory and combines them into a single DataFrame
- **Deduplication**: Removes duplicate records
- **Embedding Generation**: Uses `SentenceTransformer` (all-MiniLM-L6-v2) to generate vector embeddings from paper abstracts
- **Output**: Saves processed data with embeddings to `data_with_embeddings.pkl`

**Key Steps:**
- Import multiple data sources
- Remove duplicates
- Generate 384-dimensional embeddings for each abstract
- Store embeddings with original data

### 2. Data Classification (`data_classification.ipynb`)

- **Embedding Normalization**: Normalizes abstract embeddings using sklearn preprocessing
- **Clustering**: Applies K-Means clustering with 6 clusters to group similar papers
- **Dimensionality Reduction**: Reduces embeddings to 2D using PCA for visualization
- **Keyword Extraction**: Identifies top TF-IDF keywords for each cluster
- **Category Naming**: Assigns semantic names to clusters based on keyword analysis
- **Export**: Saves classified data by category to `Final Database.xlsx`

**Key Steps:**
- Normalize embeddings
- K-Means clustering (n_clusters=6)
- PCA visualization
- TF-IDF analysis for keyword extraction
- Cluster interpretation and naming
- Export to Excel with sanitized sheet names

## Technologies & Dependencies

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning (clustering, PCA, TF-IDF)
- **sentence-transformers** - Semantic embeddings
- **matplotlib** - Visualization
- **openpyxl/xlsxwriter** - Excel export

### Installation

```bash
pip install pandas numpy scikit-learn sentence-transformers matplotlib openpyxl xlsxwriter
```

## Cluster Categories

The pipeline identifies and categorizes papers into 6 research clusters:

1. **Concrete Strength Prediction / ML** - Machine learning applications for concrete properties
2. **Sandwich Panels & Impact Analysis** - Multi-layer composite impact studies
3. **Composite Materials & Impact** - General composite material behavior
4. **Crashworthiness / Energy Absorption** - Impact energy dissipation and safety
5. **Honeycomb Structures & Energy Absorption** - Cellular material energy absorption
6. **Reinforced Concrete & Impact** - Reinforced concrete impact resistance

## Output Files

- **Final Database.xlsx** - Classified research data with separate sheets for each cluster
  - Each sheet contains all papers categorized by cluster
  - Columns include original data plus cluster assignments

## Usage

### Running the Pipeline

1. **Data Processing**
   ```bash
   jupyter notebook data_processing.ipynb
   ```
   - Process raw datasets
   - Generate embeddings (first run takes time for embedding generation)
   - Output: `Data/data_with_embeddings.pkl`

2. **Data Classification**
   ```bash
   jupyter notebook data_classification.ipynb
   ```
   - Cluster processed data
   - Generate visualizations and keyword analysis
   - Export classified data
   - Output: `Final Database.xlsx`

### Key Parameters

- **K-Means Clusters**: 6 clusters (configurable)
- **PCA Components**: 2D visualization
- **TF-IDF Features**: Top 5,000 features
- **Embedding Model**: all-MiniLM-L6-v2 (384-dimensional)

## Visualization

The pipeline generates:
- **PCA Scatter Plot**: 2D visualization of all documents colored by cluster assignment
- Shows clustering quality and cluster separation

## Notes

- Embedding generation on first run may take significant time depending on dataset size
- Excel sheet names are automatically sanitized (max 31 characters, invalid characters removed)
- The clustering is deterministic with `random_state=42` for reproducibility

## Future Enhancements

- Add hierarchical clustering analysis
- Implement topic modeling (LDA)
- Create interactive visualizations (Plotly)
- Add cross-validation for cluster stability
- Implement document similarity search
