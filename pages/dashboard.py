import streamlit as st

from src.database.database import get_session, list_projects, list_runs

st.title("📊 Dashboard")

session = get_session()
try:
    projects = list_projects(session)

    if not projects:
        st.info("No models analyzed yet. Head to **New Analysis** to run your first evaluation.")
    else:
        # Collect all runs once to avoid duplicate queries
        project_runs = {p.id: list_runs(session, p.id) for p in projects}
        all_runs = [r for runs in project_runs.values() for r in runs]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Models tracked", len(projects))
        col2.metric("Total evaluations", len(all_runs))
        if all_runs:
            avg_acc = sum(r.accuracy for r in all_runs) / len(all_runs)
            col3.metric("Average accuracy", f"{avg_acc:.1%}")
            latest = max(all_runs, key=lambda r: r.created_at)
            col4.metric("Most recent run", latest.created_at.strftime("%Y-%m-%d"))
        else:
            col3.metric("Average accuracy", "--")
            col4.metric("Most recent run", "--")

        st.divider()
        st.subheader("Recent analyses")

        for p in projects:
            runs = project_runs[p.id]
            if not runs:
                continue
            with st.expander(f"**{p.name}** ({p.model_name}) -- {len(runs)} run(s)", expanded=False):
                for r in reversed(runs):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.write(r.created_at.strftime("%Y-%m-%d %H:%M"))
                    c2.write(f"Accuracy: {r.accuracy:.2%}")
                    c3.write(f"Precision: {r.precision:.2%}")
                    c4.write(f"Recall: {r.recall:.2%}")
                    c5.write(f"F1: {r.f1_score:.2%}")
finally:
    session.close()
