# AU17-CSV Explorer & Q&A Agent
AI Agent

📘 README.md
# CSV Explorer & Q&A Agent

A local Streamlit agent to upload a CSV, profile the data, ask questions in simple NL or SQL, and plot quick charts.

## Features
- Upload CSV (no size hard limit, depends on RAM)
- Profiling: schema, missingness, summary stats, correlations
- Q&A:
  - Simple NL patterns (average/sum/max/min/unique/filter)
  - SQL via `pandasql` (use table name `df`)
- Charts: line, bar, scatter, histogram

## Run
1. Install deps:


pip install -r requirements.txt

2. Start:


streamlit run app.py

3. Open the browser UI, upload a CSV, explore!

## Examples
- `average of price`
- `sum of sales`
- `unique values of city`
- `filter price > 100 and show top 5`
- `sql: select city, avg(price) as avg_price from df group by 1 order by avg_price desc limit 10;`

🧭 How it works (quick explanation)

tools.py

CSVTool loads the CSV and exposes a DataFrame.

ProfileTool computes schema, missingness, stats, correlations.

QueryTool parses simple natural language or runs SQL (pandasql) if input is prefixed with sql:.

ChartTool hands back relevant columns/data to be plotted in Streamlit.

agent.py

CSVAgent orchestrates tools and keeps a rolling memory (so you can see what happened).

app.py

Streamlit UI with tabs: Overview, Query, Chart, Memory.
