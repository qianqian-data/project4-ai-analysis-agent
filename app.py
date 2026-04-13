import streamlit as st
import pandas as pd
import plotly.express as px
import anthropic

# Page config
st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("🤖 AI Data Analysis Agent")
st.markdown("Upload a CSV file and let AI generate business insights for you.")
st.divider()

# Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.divider()
    
    # Chart
    st.subheader("📊 Quick Visualization")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if len(numeric_cols) >= 2:
        col_x = st.selectbox("X axis", numeric_cols)
        col_y = st.selectbox("Y axis", numeric_cols, index=1)
        fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_x} vs {col_y}")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # AI Analysis
    st.subheader("🧠 AI-Generated Insights")
    
    api_key = st.text_input("Enter your Anthropic API Key", type="password")
    
    if st.button("Generate Insights", type="primary"):
        if not api_key:
            st.warning("Please enter your API key.")
        else:
            with st.spinner("AI is analyzing your data..."):
                try:
                    summary = f"""
Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns
Columns: {list(df.columns)}
Data types: {df.dtypes.to_dict()}
Basic statistics:
{df.describe().to_string()}

First 5 rows:
{df.head().to_string()}
                    """
                    
                    client = anthropic.Anthropic(api_key=api_key)
                    
                    message = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1000,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""You are a senior data analyst. Analyze this dataset and provide business insights.

{summary}

Please provide:
1. **Dataset Overview** — what this data appears to be about
2. **Key Findings** — 3 most important patterns or insights you notice
3. **Business Recommendations** — 2-3 actionable suggestions based on the data
4. **Risks or Limitations** — any data quality issues or caveats

Be specific, concise, and business-focused."""
                            }
                        ]
                    )
                    
                    st.markdown(message.content[0].text)
                    
                except Exception as e:
                    st.error(f"Error: {e}")