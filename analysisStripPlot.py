import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_strip_plot(col):
    col.subheader("Distribution of Original vs Imputed Values")

    before_df = st.session_state.get("file", None)
    after_df = st.session_state.get("imputed_file", None)

    if before_df is None or after_df is None:
        col.warning("Missing original and imputed datasets.")
        return

    imputed_columns = before_df.columns[before_df.isnull().any() & after_df.notnull().all()]

    if imputed_columns.empty:
        col.warning("No imputed columns detected.")
        return

    selected_col = col.selectbox("Select feature", imputed_columns, key="feature_select_box_strip_plot")

    # Identify rows where values were originally missing (and thus imputed)
    imputed_indices = before_df[selected_col][before_df[selected_col].isnull()].index
    original_indices = before_df[selected_col][before_df[selected_col].notnull()].index

    original_vals = before_df.loc[original_indices, selected_col]
    imputed_vals = after_df.loc[imputed_indices, selected_col]
    total_vals = after_df[selected_col]

    comparison = pd.DataFrame({
        'Original Values': original_vals,
        'Imputed Values': imputed_vals,
        'Total (Original + Imputed)': total_vals
    }).melt(var_name='Stage', value_name='Value')

    # Create strip plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.stripplot(
        x="Stage",
        y="Value",
        data=comparison,
        palette=['skyblue', 'salmon', 'lightgreen'],
        jitter=True,
        dodge=True,
        alpha=0.7,
        ax=ax
    )

    ax.set_ylabel(selected_col)
    ax.set_title(f"Strip Plot: Original vs Imputed vs Total Values for '{selected_col}'")
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=10)

    col.pyplot(fig)
