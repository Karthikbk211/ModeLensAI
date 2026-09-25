import pandas as pd
import plotly.express as px
import streamlit as st

from src.database.database import (
    get_run,
    get_session,
    list_projects,
    list_runs,
)

st.title("🕘 History")

session = get_session()
try:
    projects = list_projects(session)

    if not projects:
        st.info("No saved analyses yet.")
        st.stop()

    project_names = {p.name: p for p in projects}
    selected_name = st.selectbox("Project", list(project_names.keys()))
    project = project_names[selected_name]
    runs = list_runs(session, project.id)

    if not runs:
        st.info("This project has no saved runs yet.")
        st.stop()

    st.subheader(f"Run history -- {project.name} ({project.model_name})")

    history_df = pd.DataFrame(
        [
            {
                "Run": i + 1,
                "Date": r.created_at.strftime("%Y-%m-%d %H:%M"),
                "Accuracy": r.accuracy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1": r.f1_score,
                "ROC-AUC": r.roc_auc,
                "Samples": r.n_samples,
                "run_id": r.id,
            }
            for i, r in enumerate(runs)
        ]
    )

    st.dataframe(
        history_df.drop(columns=["run_id"]).style.format(
            {"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1": "{:.2%}"}
        ),
        use_container_width=True,
    )

    fig = px.line(
        history_df, x="Run", y=["Accuracy", "Precision", "Recall", "F1"], markers=True,
        title="Metric trend across runs",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Inspect a run")
    run_choice = st.selectbox(
        "Select run", history_df["Run"].tolist(), format_func=lambda n: f"Run {n}"
    )
    run_id = int(history_df.loc[history_df["Run"] == run_choice, "run_id"].iloc[0])
    run = get_run(session, run_id)

    st.write(f"**Date:** {run.created_at.strftime('%Y-%m-%d %H:%M')}")
    st.write(f"**Samples evaluated:** {run.n_samples}")

    if run.failures:
        st.markdown("**Recorded failure findings**")
        for finding in run.failures:
            st.markdown(f"- `{finding.failure_type}` -- {finding.description}")

    if run.genai_summary:
        st.markdown("**GenAI summary**")
        st.write(run.genai_summary.summary)
        if run.genai_summary.recommendations:
            st.markdown("**Recommendations**")
            for rec in run.genai_summary.recommendations:
                st.markdown(f"- {rec}")
finally:
    session.close()
