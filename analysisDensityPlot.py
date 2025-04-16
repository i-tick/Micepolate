import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def render_density_plot(col):
    col.subheader("Distribution of Original vs Imputed Values")

    before_df = st.session_state.get("file", None)
    after_df = st.session_state.get("imputed_file", None)

    if before_df is None or after_df is None:
        col.warning("Missing original and imputed datasets.")
        return

    # Identify columns that had missing values and were fully imputed
    imputed_columns = before_df.columns[before_df.isnull().any() & after_df.notnull().all()]

    if imputed_columns.empty:
        col.warning("No imputed columns detected.")
        return

    selected_col = col.selectbox("Select Feature", imputed_columns, key="feature_select_box_distribution_plot")

    # Indices where values were missing (and hence imputed)
    imputed_indices = before_df[selected_col][before_df[selected_col].isnull()].index
    original_indices = before_df[selected_col][before_df[selected_col].notnull()].index

    original_vals = before_df.loc[original_indices, selected_col]
    imputed_vals = after_df.loc[imputed_indices, selected_col]
    total_vals = after_df[selected_col]

    # Plot KDE distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(original_vals, label="Original Values", shade=True, color="skyblue", ax=ax)
    sns.kdeplot(imputed_vals, label="Imputed Values", shade=True, color="salmon", ax=ax)
    sns.kdeplot(total_vals, label="Total (Original + Imputed)", shade=True, color="lightgreen", ax=ax)

    ax.set_xlabel(selected_col)
    ax.set_ylabel("Density")
    ax.set_title(f"KDE Plot: Original vs Imputed vs Total Values for '{selected_col}'")
    ax.legend()
    ax.tick_params(axis='both', labelsize=10)

    col.pyplot(fig)
