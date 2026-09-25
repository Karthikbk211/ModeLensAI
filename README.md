# ModelLens AI

**ModelLens AI** is a GenAI + RAG powered machine-learning model failure analysis assistant.

It evaluates a trained scikit-learn classification model, identifies measurable failure patterns, retrieves relevant ML troubleshooting knowledge, and sends both the observed evidence and retrieved context to an LLM for an evidence-grounded diagnosis.

## Architecture

```text
Model + Dataset
      ↓
ML Evaluation
      ↓
Failure Analysis
      ↓
Retrieval Query
      ↓
Knowledge Base → Embeddings → Similarity Retrieval
      ↓
Retrieved ML Context
      ↓
LLM / GenAI
      ↓
Failure Summary + Possible Factors + Recommendations
      ↓
PostgreSQL
      ↓
Analysis History
```

## Stack

- **UI:** Streamlit
- **ML:** scikit-learn, pandas, NumPy
- **Visualization:** Plotly
- **RAG:** TF-IDF vector embeddings + cosine-similarity retrieval
- **Knowledge Base:** curated ML evaluation/failure-analysis documents
- **LLM:** Google Gemini API (gemini-3.8-flash, free tier)
- **Database:** PostgreSQL in deployment (Neon.tech); SQLite fallback for local development
- **ORM:** SQLAlchemy
- **Deployment:** Streamlit Community Cloud or another Streamlit-compatible host

## RAG workflow

The application ships with a small knowledge base covering classification metrics, class imbalance, confusion matrices, overfitting, data leakage, feature quality, evaluation practices, and model improvement.

At analysis time ModelLens creates a query from the **actual measured model failures**, embeds the query, retrieves the most relevant knowledge chunks, and supplies those chunks to the LLM along with the computed metrics and failure analysis.

The LLM is instructed to distinguish measured evidence from possible explanations and recommendations.

## Database

The database stores:

- Projects
- Evaluation runs
- Failure-analysis findings
- GenAI summaries
- RAG knowledge chunks and their embeddings

Set `DATABASE_URL` to a hosted PostgreSQL connection string for deployment. If it is absent, the application uses a local SQLite database so development works without a database server.

## Getting started

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Use **New Analysis → Use sample model** for the bundled demonstration.

## Secrets

For local development, use `.streamlit/secrets.toml`:

```toml
DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require"
GEMINI_API_KEY = "YOUR_API_KEY"
```

For Streamlit deployment, put the same values in the app's Secrets settings. Never commit real credentials.

## Current scope

The prototype supports fitted scikit-learn classification models that expose `.predict()`. Regression and other frameworks can be added later.

**Security note:** uploaded pickle files are executable Python objects and should only be accepted from trusted sources. A production multi-tenant service should sandbox uploads or use a safer serialization format.
