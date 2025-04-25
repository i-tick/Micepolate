from analysisStripPlot import render_strip_plot
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from missingnessScatter import *
from analysisBoxPlot import *
from analysisDensityPlot import *

def analysis_page_content():
    tab1, tab2, tab3 = st.tabs(["Imputation Analysis", "Interpolation Analysis", "Compare with Original Data"])
    with tab1:

        st.header("Imputation Analysis")
        selected_algorithm = st.selectbox("Select Algorithm", ["MICE", "BART"], key="algorithm_selectbox")

        # Hardcoded file selection based on the selected algorithm
        if selected_algorithm == "MICE":
            imputed_file = st.session_state["imputed_file"]
            st.success("MICE imputed file loaded.")
        elif selected_algorithm == "BART":
            imputed_file  = st.session_state["imputed_file_bart"]
            st.success("BART imputed file loaded.")

        col1, col2 = st.columns([1, 1])

        with col1:
            missingness_statistics(col1)
            # render_box_plot(col1)
            # render_strip_plot(col1)
            download_imputed_file = st.button("Download Imputed File")
            if download_imputed_file:
                if "imputed_file" in st.session_state:
                    imputed_df = st.session_state["imputed_file"]
                    imputed_df.to_csv("imputed_data.csv", index=False)
                    st.success("Imputed file downloaded successfully.")
                else:
                    st.error("No imputed file available for download.")

        with col2:
            render_scatter_plots_with_scores(col2, imputed_file)
            # render_density_plot(col2)

    with tab2:
        st.header("Comarison of Interpolation Methods")
        st.write(
            """
            This section is dedicated to analyzing interpolation methods. 
            Use this tab to evaluate how different interpolation techniques 
            perform on your data and their impact on downstream analysis.
            """
        )

        before_df = st.session_state.get("file", None)
        after_df = st.session_state.get("imputed_file", None)
        imputed_columns = before_df.columns[before_df.isnull().any() & after_df.notnull().all()]
        # X-axis selection in first column
        selected_x_col = st.selectbox("Select X-axis column", before_df.columns, key=f"x_axis_selectbox")

        # Y-axis selection (imputed columns only) in second column
        selected_y_col = st.selectbox("Select Y-axis column (Imputed)", imputed_columns, key=f"y_axis_selectbox")

        col1,col2 = st.columns([1,1])
        with col1:
            render_scatter_plots_with_file(col1, st.session_state.get('imputed_file', None),selected_x_col, selected_y_col, "Mice")
        with col2:
            render_scatter_plots_with_file(col1, st.session_state.get('imputed_file_bart', None),selected_x_col, selected_y_col, "BART")


    with tab3:
        st.subheader("Comparison of Original vs Imputed Values")

        # Check if the original file is already uploaded and saved in session state
        if "original_file" not in st.session_state:
            uploaded_file = st.file_uploader("Upload the original CSV file", type=['csv'])
            if uploaded_file is not None:
                st.session_state["original_file"] = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully.")
                st.rerun()
        else:
            st.info("Original file already uploaded.")
            original_df = st.session_state["original_file"]
            before_df = st.session_state.get("file", None)
            imputed_file_options = {
                "MICE": st.session_state.get("imputed_file", None),
                "BART": st.session_state.get("imputed_file_bart", None)
            }
            selected_imputed_file = st.selectbox("Select Imputed File", list(imputed_file_options.keys()))
            after_df = imputed_file_options[selected_imputed_file]

            if before_df is not None and after_df is not None:
                # Let user select the column to analyze
                selected_column = st.selectbox("Select column to analyze", before_df.columns)

                # Identify rows with null values in the selected column
                null_indices = before_df[before_df[selected_column].isnull()].index
                if len(null_indices) > 0:
                    # Create three columns for the plots
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Plot 1: Original vs Imputed Values with connecting lines
                        fig1, ax1 = plt.subplots(figsize=(10, 6))

                        # Create the plot using plotly for interactivity
                        import plotly.graph_objects as go

                        # Create traces for original and imputed values
                        trace1 = go.Scatter(
                            x=null_indices,
                            y=original_df.loc[null_indices, selected_column],
                            mode='markers',
                            name='Original Values',
                            marker=dict(symbol='circle', size=10, color='orange'),
                            hoverinfo='text',
                            hovertext=[f'Index: {idx}<br>Value: {val:.2f}' for idx, val in
                                       zip(null_indices, original_df.loc[null_indices, selected_column])]
                        )
                        trace2 = go.Scatter(
                            x=null_indices,
                            y=after_df.loc[null_indices, selected_column],
                            mode='markers',
                            name='Imputed Values',
                            marker=dict(symbol='x', size=10),
                            hoverinfo='text',
                            hovertext=[f'Index: {idx}<br>Value: {val:.2f}' for idx, val in
                                       zip(null_indices, after_df.loc[null_indices, selected_column])]
                        )

                        # Create connecting lines
                        lines = []
                        for idx in null_indices:
                            lines.append(
                                go.Scatter(
                                    x=[idx, idx],
                                    y=[original_df.loc[idx, selected_column], after_df.loc[idx, selected_column]],
                                    mode='lines',
                                    line=dict(color='gray', dash='dash', width=1),
                                    showlegend=False,
                                    hoverinfo='skip'
                                )
                            )

                        # Combine all traces
                        data = [trace1, trace2] + lines

                        # Create layout
                        layout = go.Layout(
                            title='Original vs Imputed Values',
                            xaxis=dict(title='Index'),
                            yaxis=dict(title=selected_column),
                            hovermode='closest',
                            showlegend=True
                        )

                        # Create figure
                        fig = go.Figure(data=data, layout=layout)

                        # Add hover effects
                        fig.update_traces(
                            hoverlabel=dict(bgcolor="white", font_size=12),
                            selector=dict(type='scatter')
                        )

                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Plot 2: Distribution of differences
                        differences = abs(after_df.loc[null_indices, selected_column] - original_df.loc[
                            null_indices, selected_column])
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        ax2.hist(differences, bins=20)
                        ax2.set_xlabel('Absolute Difference')
                        ax2.set_ylabel('Frequency')
                        ax2.set_title('Distribution of Differences')
                        st.pyplot(fig2)

                    with col3:
                        # Plot 3: Imputation shift per data point
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        ax3.scatter(original_df.loc[null_indices, selected_column],
                                    after_df.loc[null_indices, selected_column] - original_df.loc[
                                        null_indices, selected_column])
                        ax3.set_xlabel(f"Original {selected_column}")
                        ax3.set_ylabel(f"Imputed {selected_column} Shift")
                        ax3.set_title("Imputation Shift per Data Point")
                        st.pyplot(fig3)

                    # Display summary statistics
                    st.subheader("Summary Statistics of Differences")
                    st.dataframe(differences.describe())
                else:
                    st.info(f"No null values found in the {selected_column} column.")
            else:
                st.warning("Please ensure both original and imputed datasets are available in the session state.")


def missingness_statistics(col):
    if "NaN-mask" in st.session_state:
        nan_mask = st.session_state["NaN-mask"]
        missing_stats = nan_mask.sum(axis=0).reset_index()
        missing_stats.columns = ["Column", "Number of NaNs"]
        missing_stats["Percentage"] = (
                (missing_stats["Number of NaNs"] / len(nan_mask)) * 100
        ).round(2)
        col.subheader("Missingness Statistics")
        col.dataframe(missing_stats)
    else:
        col.warning("The NaN-mask is not available. Please ensure it is set in session state.")


def render_scatter_plots_with_scores(col, file=None):
    if 'imputed_file' in st.session_state:
        col.subheader('Plot of Imputated Values on Dataset')
        before_df = st.session_state.get("file", None)
        after_df = file
        imputed_columns = before_df.columns[before_df.isnull().any() & after_df.notnull().all()]

        col1, col2 = col.columns(2)

        # X-axis selection in first column
        with col1:
            selected_x_col = st.selectbox("Select X-axis column", before_df.columns)

        # Y-axis selection (imputed columns only) in second column
        with col2:
            selected_y_col = st.selectbox("Select Y-axis column (Imputed)", imputed_columns)

        # Compute metrics for quality assessment
        quality_metrics = {}

        fig, ax = plt.subplots(figsize=(10, 6))

        original_mask = ~before_df[selected_y_col].isnull()
        sns.scatterplot(x=before_df[selected_x_col][original_mask], y=before_df[selected_y_col][original_mask],
                        label='Rest Values', color='blue', ax=ax)

        imputed_mask = before_df[selected_y_col].isnull()
        sns.scatterplot(x=after_df[selected_x_col][imputed_mask], y=after_df[selected_y_col][imputed_mask],
                        label='Imputed Values', color='orange', ax=ax)

        plt.title(f"Scatter Plot for {selected_y_col} (Before and After Imputation)")
        plt.xlabel(selected_x_col)  # Updated x-axis label
        plt.ylabel(selected_y_col)
        plt.legend()
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        for col in imputed_columns:
            before_values = before_df[selected_y_col]
            after_values = after_df[selected_y_col]

            # Count of imputed values
            num_imputed = before_values.isnull().sum()

            # Mean absolute change in imputed values (only for numeric columns)
            if np.issubdtype(after_values.dtype, np.number):
                mean_change = np.abs(before_values.fillna(0) - after_values).mean()
            else:
                mean_change = "N/A (categorical column)"

            # Store results
            quality_metrics[col] = {
                "Imputed Count": num_imputed,
                "Mean Change": mean_change
            }
        # Convert to DataFrame for display
        quality_df = pd.DataFrame.from_dict(quality_metrics, orient="index")

        # Print or return the quality DataFrame
        # st.write("Imputation Quality Assessment", quality_df)

    else:
        col.warning('Error in Imputation')

def render_scatter_plots_with_file(col, file, selected_x_col=None, selected_y_col=None, algorithm=None):
    if file is not None:
        before_df = st.session_state.get("file", None)
        after_df = file
        imputed_columns = before_df.columns[before_df.isnull().any() & after_df.notnull().all()]

        # Compute metrics for quality assessment
        quality_metrics = {}

        fig, ax = plt.subplots(figsize=(10, 6))

        original_mask = ~before_df[selected_y_col].isnull()
        sns.scatterplot(x=before_df[selected_x_col][original_mask], y=before_df[selected_y_col][original_mask],
                        label='Rest Values', color='blue', ax=ax)

        imputed_mask = before_df[selected_y_col].isnull()
        sns.scatterplot(x=after_df[selected_x_col][imputed_mask], y=after_df[selected_y_col][imputed_mask],
                        label='Imputed Values', color='orange', ax=ax)

        plt.title(f"Scatter Plot for {selected_y_col} (Before and After Imputation) - {algorithm}")
        plt.xlabel(selected_x_col)  # Updated x-axis label
        plt.ylabel(selected_y_col)
        plt.legend()
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        for col in imputed_columns:
            before_values = before_df[selected_y_col]
            after_values = after_df[selected_y_col]

            # Count of imputed values
            num_imputed = before_values.isnull().sum()

            # Mean absolute change in imputed values (only for numeric columns)
            if np.issubdtype(after_values.dtype, np.number):
                mean_change = np.abs(before_values.fillna(0) - after_values).mean()
            else:
                mean_change = "N/A (categorical column)"

            # Store results
            quality_metrics[col] = {
                "Imputed Count": num_imputed,
                "Mean Change": mean_change
            }
        # Convert to DataFrame for display
        quality_df = pd.DataFrame.from_dict(quality_metrics, orient="index")

        # Print or return the quality DataFrame
        # st.write("Imputation Quality Assessment", quality_df)

    else:
        col.warning('Error in Imputation')