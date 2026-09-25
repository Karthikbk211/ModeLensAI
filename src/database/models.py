"""
SQLAlchemy ORM models for ModelLens AI.

Tables: projects -> evaluation_runs -> failure_analyses -> recommendations
(a single-user local tool doesn't need a users table yet; add one later
if auth is introduced).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    model_name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    runs: Mapped[list["EvaluationRun"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    accuracy: Mapped[float] = mapped_column(Float)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1_score: Mapped[float] = mapped_column(Float)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_samples: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="runs")
    failures: Mapped[list["FailureAnalysis"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    genai_summary: Mapped["GenAISummary"] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )


class FailureAnalysis(Base):
    __tablename__ = "failure_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"))
    failure_type: Mapped[str] = mapped_column(String(50))  # class | misclassification | feature
    affected_class: Mapped[str | None] = mapped_column(String(255), nullable=True)
    frequency: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped["EvaluationRun"] = relationship(back_populates="failures")


class GenAISummary(Base):
    __tablename__ = "genai_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"))
    summary: Mapped[str] = mapped_column(Text)
    contributing_factors: Mapped[list] = mapped_column(JSON, default=list)
    recommendations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    run: Mapped["EvaluationRun"] = relationship(back_populates="genai_summary")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    # JSON-encoded normalized embedding. Stored in PostgreSQL for persistence;
    # cosine similarity is calculated by the retrieval layer.
    embedding: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
