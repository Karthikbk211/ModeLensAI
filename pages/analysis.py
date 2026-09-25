import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.database.database import get_or_create_project, get_session, save_run
from src.evaluation.confusion import confusion_matrix_figure
from src.evaluation.failure_analysis import get_misclassified_samples, run_failure_analysis
from src.evaluation.metrics import evaluate_model
from src.genai.analyzer import explain_failures
from src.rag.retriever import retrieve
from src.utils.preprocessing import ModelLensError, load_dataset, load_model, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.title("🧪 New Analysis")
st.caption("Evaluate a model, retrieve relevant ML knowledge, and generate an evidence-grounded failure diagnosis.")

use_sample = st.toggle("Use sample model (skip uploads)", value=False)

model = None
df = None
target_column = None
project_name = None
model_name = None

if use_sample:
    st.caption("Loads the bundled Breast Cancer Wisconsin Random Forest sample.")
    sample_model_path = PROJECT_ROOT / "sample_data" / "sample_model.pkl"
    sample_data_path = PROJECT_ROOT / "sample_data" / "sample_dataset.csv"
    with open(sample_model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv(sample_data_path)
    target_column = "target"
    project_name = "Sample: Breast Cancer Classifier"
    model_name = "RandomForestClassifier"
    st.dataframe(df.head(), use_container_width=True)
else:
    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("Upload model (.pkl)", type=["pkl"])
    with col2:
        data_file = st.file_uploader("Upload dataset (.csv)", type=["csv"])

    if model_file and data_file:
        try:
            model = load_model(model_file.getvalue())
            df = load_dataset(data_file.getvalue(), data_file.name)
        except ModelLensError as e:
            st.error(str(e))
            st.stop()

        st.dataframe(df.head(), use_container_width=True)
        target_column = st.selectbox("Target column", df.columns.tolist())
        project_name = st.text_input(
            "Project / model display name",
            value=data_file.name.replace(".csv", ""),
        )
        model_name = type(model).__name__

run_genai = st.checkbox("Generate GenAI explanation", value=True)

if model is not None and df is not None and target_column:
    if st.button("Analyze Model", type="primary"):
        try:
            X, y = split_features_target(df, target_column)
        except ModelLensError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Evaluating model..."):
            try:
                result = evaluate_model(model, X, y)
            except Exception as e:
                st.error(f"Model evaluation failed: {e}. Check that the dataset columns match the model's expected features.")
                st.stop()

        with st.spinner("Running failure analysis..."):
            failure = run_failure_analysis(result, X)

        # Build a retrieval query from the actual measured findings.
        flagged = [x for x in failure["class_failures"] if x.get("flagged")]
        mis = failure["top_misclassifications"][:3]
        feature_names = [x["feature"] for x in failure["feature_errors"][:4]]
        query = (
            f"Model {model_name}. Accuracy {result.accuracy:.3f}, recall {result.recall:.3f}, "
            f"F1 {result.f1:.3f}. Flagged class failures: {flagged}. "
            f"Top misclassifications: {mis}. Error-concentrated features: {feature_names}. "
            "What ML evaluation and failure-analysis practices are relevant?"
        )

        session = get_session()
        try:
            with st.spinner("Retrieving relevant ML knowledge..."):
                retrieved = retrieve(session, query, top_k=4)
        finally:
            session.close()

        genai_result = None
        if run_genai:
            with st.spinner("Generating evidence-grounded diagnosis..."):
                genai_result = explain_failures(
                    result.to_summary_dict(),
                    failure,
                    model_name=model_name,
                    n_samples=len(X),
                    retrieved_context=retrieved,
                )

        st.session_state["analysis_result"] = {
            "result": result,
            "failure": failure,
            "X": X,
            "model_name": model_name,
            "project_name": project_name,
            "n_samples": len(X),
            "retrieved": retrieved,
            "genai_result": genai_result,
        }

state = st.session_state.get("analysis_result")

if state:
    result = state["result"]
    failure = state["failure"]
    X = state["X"]
    retrieved = state["retrieved"]
    genai_result = state["genai_result"]

    st.header("Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{result.accuracy:.2%}")
    c2.metric("Precision", f"{result.precision:.2%}")
    c3.metric("Recall", f"{result.recall:.2%}")
    c4.metric("F1 Score", f"{result.f1:.2%}")
    c5.metric("ROC-AUC", f"{result.roc_auc:.2%}" if result.roc_auc is not None else "N/A")

    st.subheader("Confusion Matrix")
    st.plotly_chart(
        confusion_matrix_figure(result.confusion, result.class_labels),
        use_container_width=True,
    )

    st.header("⭐ Failure Patterns")
    st.subheader("Class-level failures")
    st.dataframe(pd.DataFrame(failure["class_failures"]), use_container_width=True)

    st.subheader("Top misclassifications")
    if failure["top_misclassifications"]:
        st.dataframe(pd.DataFrame(failure["top_misclassifications"]), use_container_width=True)
    else:
        st.write("No misclassifications found -- perfect predictions on this set.")

    st.subheader("Feature-based error concentration")
    for fe in failure["feature_errors"][:5]:
        with st.expander(f"**{fe['feature']}** (max deviation {fe['max_deviation']:.2%})"):
            st.dataframe(pd.DataFrame(fe["bins"]), use_container_width=True)

    with st.expander("Misclassified samples"):
        st.dataframe(
            get_misclassified_samples(X, result.y_true, result.y_pred),
            use_container_width=True,
        )

    st.header("📚 RAG Context")
    st.caption("These knowledge-base chunks were retrieved from the ML troubleshooting corpus and passed to the LLM.")
    for item in retrieved:
        with st.expander(f"{item['source']}  •  similarity {item['score']:.3f}"):
            st.write(item["content"])

    if genai_result:
        st.header("🤖 GenAI Diagnosis")
        st.markdown(f"**Model Failure Summary**\n\n{genai_result['summary']}")
        if genai_result["contributing_factors"]:
            st.markdown("**Possible contributing factors**")
            for factor in genai_result["contributing_factors"]:
                st.markdown(f"- {factor}")
        if genai_result["recommendations"]:
            st.markdown("**Suggested areas to investigate**")
            for rec in genai_result["recommendations"]:
                st.markdown(f"- {rec}")

    st.divider()
    if st.button("💾 Save Analysis"):
        session = get_session()
        try:
            project = get_or_create_project(
                session, state["project_name"], state["model_name"]
            )
            save_run(
                session,
                project,
                result.to_summary_dict(),
                n_samples=state["n_samples"],
                failure_analysis=failure,
                genai_result=genai_result,
            )
        finally:
            session.close()
        st.success(f"Saved run for project '{state['project_name']}'. See it under History.")
