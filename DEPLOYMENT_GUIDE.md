# ModelLens AI — Submission Deployment Guide

## 1. Push the updated project to GitHub

Replace the files in your existing ModelLens repository with this project, then commit and push.

Important folders/files that must be included:
- app.py
- pages/
- src/
- knowledge_base/
- sample_data/
- requirements.txt
- packages.txt
- .streamlit/secrets.toml.example
- README.md

Do NOT upload `.streamlit/secrets.toml` with real credentials.

## 2. Create the PostgreSQL database

Use your Neon.tech project (or Supabase, or any hosted PostgreSQL provider).

Copy the PostgreSQL connection string from your dashboard. It should look like:

    postgresql+psycopg2://USER:PASSWORD@HOST.neon.tech:5432/DATABASE?sslmode=require

The application uses SQLAlchemy, so the connection is supplied via `DATABASE_URL`.

The application automatically creates its tables on startup.

## 3. Get a Google Gemini API key

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

The app reads it from:

    GEMINI_API_KEY = "your-key"

Without the key, the ML evaluation and RAG retrieval still work, but the LLM diagnosis will show the fallback message.

## 4. Deploy with Streamlit Community Cloud

Go to:

https://share.streamlit.io/

Connect your GitHub account, choose the ModelLens repository, select the branch, and use:

    app.py

as the entrypoint.

During deployment, open **Advanced settings → Secrets** and paste:

    DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@HOST.neon.tech:5432/DATABASE?sslmode=require"
    GEMINI_API_KEY = "your-gemini-api-key"

Do not put the credentials into GitHub.

## 5. First test after deployment

Open the deployed application.

Go to:

    New Analysis
    → Use sample model
    → Analyze Model

You should see:

1. Accuracy / Precision / Recall / F1 / ROC-AUC
2. Confusion matrix
3. Failure patterns
4. RAG Context
5. GenAI Diagnosis
6. Save Analysis

Then open:

    History

and confirm that the saved run appears.

## 6. What to demonstrate during submission

The easiest demo flow is:

    Upload/use sample model
    → Analyze
    → show ML metrics
    → show a failure pattern
    → open RAG Context
    → show retrieved ML knowledge
    → show GenAI Diagnosis
    → Save Analysis
    → open History
    → show the stored run

This demonstrates the complete chain:

    ML → Failure Analysis → RAG → LLM → PostgreSQL → Streamlit Deployment.

## 7. If deployment fails

Open the deployed app's **Manage app / Cloud logs** and copy the error message.

Check for common issues:
- Missing `packages.txt` (needed for `psycopg2-binary` on Streamlit Cloud)
- Wrong secrets format (must use TOML `key = "value"` syntax)
- Database connection string missing `sslmode=require` for Neon.tech

## 8. Important security note

Never commit:
- Gemini API keys
- PostgreSQL passwords
- `.streamlit/secrets.toml`

The repository only contains `.streamlit/secrets.toml.example`.
