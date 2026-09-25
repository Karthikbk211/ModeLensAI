"""
Database engine, session management, and persistence helpers.

Reads DATABASE_URL from the environment. Defaults to a local SQLite
file so the app runs out of the box without a Postgres server -- swap
in a real Postgres URL for production:

    postgresql+psycopg2://user:password@host:5432/modelens

Set it in Streamlit secrets or your shell:
    export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/modelens"
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, EvaluationRun, FailureAnalysis, GenAISummary, Project
from src.utils.preprocessing import get_config

DEFAULT_SQLITE_URL = "sqlite:///modelens.db"
DATABASE_URL = get_config("DATABASE_URL", DEFAULT_SQLITE_URL)

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables and seed the RAG knowledge base if needed."""
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        from src.rag.retriever import ingest_knowledge_base
        ingest_knowledge_base(session)
    except Exception:
        # Database should still initialize even if the embedding model is
        # temporarily unavailable. The app can continue with ML analysis.
        pass
    finally:
        session.close()


def get_session() -> Session:
    return SessionLocal()


# ---------------------------------------------------------------------
# Persistence helpers -- these wrap the raw ORM calls so app/page code
# never has to touch SQLAlchemy directly.
# ---------------------------------------------------------------------


def get_or_create_project(session: Session, name: str, model_name: str) -> Project:
    project = session.query(Project).filter_by(name=name).first()
    if project is None:
        project = Project(name=name, model_name=model_name)
        session.add(project)
        session.commit()
        session.refresh(project)
    return project


def save_run(
    session: Session,
    project: Project,
    metrics: dict,
    n_samples: int,
    failure_analysis: dict,
    genai_result: dict | None = None,
) -> EvaluationRun:
    run = EvaluationRun(
        project_id=project.id,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        roc_auc=metrics.get("roc_auc"),
        n_samples=n_samples,
    )
    session.add(run)
    session.flush()  # get run.id before adding children

    for cf in failure_analysis.get("class_failures", []):
        if cf.get("flagged"):
            gap = cf.get("gap_vs_average")
            gap_str = f"{gap:.2%} below average" if gap is not None else "below average"
            session.add(
                FailureAnalysis(
                    run_id=run.id,
                    failure_type="class",
                    affected_class=cf["class"],
                    frequency=None,
                    description=f"Class {cf['class']} accuracy {cf['accuracy']:.2%}, {gap_str}",
                    details=cf,
                )
            )

    for mc in failure_analysis.get("top_misclassifications", []):
        session.add(
            FailureAnalysis(
                run_id=run.id,
                failure_type="misclassification",
                affected_class=f"{mc['actual']} -> {mc['predicted']}",
                frequency=mc["count"],
                description=f"{mc['count']} samples of class {mc['actual']} "
                f"predicted as {mc['predicted']}",
                details=mc,
            )
        )

    for fe in failure_analysis.get("feature_errors", []):
        session.add(
            FailureAnalysis(
                run_id=run.id,
                failure_type="feature",
                affected_class=fe["feature"],
                frequency=None,
                description=f"Feature '{fe['feature']}' shows uneven error rates across its range",
                details=fe,
            )
        )

    if genai_result:
        session.add(
            GenAISummary(
                run_id=run.id,
                summary=genai_result.get("summary", ""),
                contributing_factors=genai_result.get("contributing_factors", []),
                recommendations=genai_result.get("recommendations", []),
            )
        )

    session.commit()
    session.refresh(run)
    return run


def list_projects(session: Session) -> list[Project]:
    return session.query(Project).order_by(Project.created_at.desc()).all()


def list_runs(session: Session, project_id: int) -> list[EvaluationRun]:
    return (
        session.query(EvaluationRun)
        .filter_by(project_id=project_id)
        .order_by(EvaluationRun.created_at.asc())
        .all()
    )


def get_run(session: Session, run_id: int) -> EvaluationRun | None:
    return session.get(EvaluationRun, run_id)
