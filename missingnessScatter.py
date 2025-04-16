import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def render_missingness_scatter_plot():
    # Check if the necessary data is available in session state
    if "file" not in st.session_state or "NaN-mask" not in st.session_state:
        st.error("Please ensure both 'file' and 'NaN-mask' are available in session state.")
        return
    
    target_df = st.session_state["file"]
    nan_mask = st.session_state["NaN-mask"]
    
    # Ensure that the columns in the file and NaN-mask match
    if not target_df.columns.equals(nan_mask.columns):
        st.error("The columns in 'file' and 'NaN-mask' do not match.")
        return

    # Get the list of numerical columns for selecting x and y
    numerical_columns = target_df.select_dtypes(include=[float, int]).columns.tolist()

    # Get the first three columns from the DataFrame (if they exist) for default dropdown values
    default_x = numerical_columns[0] if len(numerical_columns) > 0 else None
    default_y = numerical_columns[1] if len(numerical_columns) > 1 else None
    default_nan_col = nan_mask.columns[2] if len(nan_mask.columns) > 2 else None
    
    col1, col2 = st.columns([1,4])

    with col1:
    # Dropdown menus for selecting x, y, and target (nan_col) with default values
        col_x = st.selectbox("Select X column", numerical_columns, index=0 if default_x else 0)
        col_y = st.selectbox("Select Y column", numerical_columns, index=1 if default_y else 1)
        nan_col = st.selectbox("Select column for missingness", nan_mask.columns, index=2 if default_nan_col else 0)

    # Once all 3 columns are chosen, render the scatter plot and score
    if col_x and col_y and nan_col:
        with col2:
            score = calculate_score(col_x, col_y, nan_col, target_df, nan_mask)
            score = f"Score: {score:.4f}"
            # st.write(f"#### Scatter Plot of {col_x} vs {col_y} (Colored by Missingness in {nan_col})")
            ScatterPlotByMissingness(col_x, col_y, nan_col, target_df, nan_mask, score)
            
            # Calculate and display the score
            

def ScatterPlotByMissingness(col_x, col_y, nan_col, target_df, nan_mask, score):
    colors = nan_mask[nan_col].apply(lambda x: 'red' if x else 'blue')
    
    plt.figure(figsize=(6, 4))
    plt.scatter(target_df[col_x], target_df[col_y], c=colors, alpha=0.6)
    plt.title(f"Scatter Plot of {col_x} vs {col_y} (Colored by {nan_col}) Score: {score}")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)

def calculate_score(col_x, col_y, nan_col, target_df, nan_mask):
    # Separate the points where missingness is True and False
    red_points = target_df[nan_mask[nan_col]].loc[:, [col_x, col_y]]
    blue_points = target_df[~nan_mask[nan_col]].loc[:, [col_x, col_y]]

    # Calculate centroids (mean values) for both red and blue groups
    centroid_red = red_points.mean().values
    centroid_blue = blue_points.mean().values

    # Calculate the Euclidean distance between the centroids
    centroid_distance = np.linalg.norm(centroid_red - centroid_blue)

    # Calculate variances for both groups
    var_red = red_points.var().values
    var_blue = blue_points.var().values

    # Score formula: Centroid distance divided by the square root of the product of variances
    score = centroid_distance / np.sqrt(np.prod(var_red) * np.prod(var_blue))
    
    return score
