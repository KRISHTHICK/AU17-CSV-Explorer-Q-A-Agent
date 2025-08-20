import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from agent import CSVAgent

st.set_page_config(page_title="CSV Explorer & Q&A Agent", page_icon="📊", layout="wide")
st.title("📊 CSV Explorer & Q&A Agent")

if "agent" not in st.session_state:
    st.session_state.agent = CSVAgent()

agent = st.session_state.agent

# --- Sidebar: Upload CSV ---
st.sidebar.header("1) Upload CSV")
file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
read_kwargs = {}
sep = st.sidebar.text_input("Delimiter (optional)", value="")
if sep:
    read_kwargs["sep"] = sep

if file is not None and st.sidebar.button("Load CSV"):
    msg = agent.load_csv(file.read(), file.name, **read_kwargs)
    st.sidebar.success(msg)

# --- Tabs ---
tab_overview, tab_query, tab_chart, tab_memory = st.tabs(["Overview", "Query", "Chart", "Memory"])

with tab_overview:
    st.subheader("Dataset Preview")
    try:
        st.dataframe(agent.head(10))
    except Exception as e:
        st.info("Upload and load a CSV to begin.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Schema")
        try:
            st.dataframe(agent.profile_schema())
        except Exception as e:
            st.caption("—")

    with col2:
        st.subheader("Missingness")
        try:
            st.dataframe(agent.profile_missing())
        except Exception as e:
            st.caption("—")

    st.subheader("Summary Stats")
    try:
        st.dataframe(agent.profile_stats())
    except Exception as e:
        st.caption("—")

    st.subheader("Correlations (numeric)")
    try:
        corr = agent.profile_corr()
        if isinstance(corr, pd.DataFrame) and not corr.empty and "note" not in corr.columns:
            st.dataframe(corr)
        else:
            st.write(corr)
    except Exception as e:
        st.caption("—")

with tab_query:
    st.subheader("Ask a Question")
    st.caption("Examples: `average of price`, `sum of sales`, `count rows`, `unique values of city`, `filter price > 100 and show top 5`, or use SQL like `sql: select city, count(*) c from df group by 1 order by c desc limit 5`.")
    q = st.text_input("Your question")
    if st.button("Run"):
        if not q.strip():
            st.warning("Enter a question.")
        else:
            try:
                res = agent.ask(q)
                if isinstance(res, pd.DataFrame):
                    st.dataframe(res)
                else:
                    st.write(res)
            except Exception as e:
                st.error(str(e))

with tab_chart:
    st.subheader("Quick Charts")
    cols = agent.available_columns()
    if not cols:
        st.info("Load a CSV to configure charts.")
    else:
        kind = st.selectbox("Chart type", ["line", "bar", "scatter", "hist"])
        x = st.selectbox("X axis", cols)
        y = None
        if kind in ("line", "bar", "scatter"):
            y = st.selectbox("Y axis (optional for line/bar; required for scatter)", ["(none)"] + cols)
            if y == "(none)":
                y = None

        if st.button("Generate Chart"):
            try:
                data = agent.chart_data(x, y, kind)
                fig, ax = plt.subplots()
                if kind == "hist":
                    ax.hist(data[x].dropna())
                    ax.set_xlabel(x)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Histogram of {x}")
                elif kind == "scatter" and y is not None:
                    ax.scatter(data[x], data[y])
                    ax.set_xlabel(x); ax.set_ylabel(y)
                    ax.set_title(f"Scatter: {x} vs {y}")
                elif kind == "bar":
                    if y is not None:
                        ax.bar(data[x].astype(str), data[y])
                        ax.set_xlabel(x); ax.set_ylabel(y)
                        ax.set_title(f"Bar: {y} by {x}")
                        plt.xticks(rotation=45, ha="right")
                    else:
                        counts = data[x].astype(str).value_counts().head(30)
                        ax.bar(counts.index, counts.values)
                        ax.set_xlabel(x); ax.set_ylabel("Count")
                        ax.set_title(f"Bar: Count by {x}")
                        plt.xticks(rotation=45, ha="right")
                else:  # line
                    if y is not None:
                        ax.plot(data[y].values)
                        ax.set_xlabel("Index"); ax.set_ylabel(y)
                        ax.set_title(f"Line: {y}")
                    else:
                        ax.plot(pd.Series(range(len(data))), data[x].values)
                        ax.set_xlabel("Index"); ax.set_ylabel(x)
                        ax.set_title(f"Line: {x}")
                st.pyplot(fig)
            except Exception as e:
                st.error(str(e))

with tab_memory:
    st.subheader("Conversation / Action Memory")
    if agent.memory:
        for m in agent.memory:
            st.write(m)
    else:
        st.caption("(empty)")
