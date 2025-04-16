import pandas as pd
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def plot_stacked_bar_graph():
    # Ensure the file is in session state
    if "file" not in st.session_state:
        st.error("Please upload a file first.")
        return

    df = st.session_state["file"]

    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    if len(categorical_columns) == 0:
        st.error("No categorical columns found in the dataframe.")
        return

    percentages = {}
    for col in categorical_columns:
        percentages[col] = df[col].value_counts(normalize=True)

    fig = go.Figure()

    categories = list(percentages.keys())
    category_data = []

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for col in categories:
        category_proportions = percentages[col]
        category_data.append(category_proportions)
        
    for i, category_proportion in enumerate(category_data):
        for j, category in enumerate(category_proportion.index):
            fig.add_trace(go.Bar(
                x=[categories[i]],  # x position corresponds to column name
                y=[category_proportion[category] * 100],  # Manually multiply by 100 to get percentage
                name=category,  # Set the name of the trace to the category label
                marker_color=colors[j % len(colors)],  # Assign a color to each category (cycling through colors)
                hovertemplate='%{x}: ' + category + ' - %{y:.2f}%',  # Hover will show column and percentage
            ))

    # Update layout to remove the legend and display hover information
    fig.update_layout(
        barmode='stack',  # Stack bars
        xaxis_title='Categorical Columns',  # x-axis title
        yaxis_title='Proportion',  # y-axis title
        showlegend=False,  # Do not show the legend
        hovermode='closest',  # Show hover information
        title='Proportions of Categories by Column'  # Title of the chart
    )

    st.plotly_chart(fig)
