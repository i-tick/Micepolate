import streamlit as st
import plotly.graph_objects as go
import pandas as pd


# @st.cache_data
def load_histogram(data, column_name):
    """
    Renders a histogram for column data distribution, a separate bar for NaN counts,
    and (if available) overlays imputed values from a separate dataset.
    """

    # Check if 'imputed_file' exists in session state
    has_imputed = "imputed_file" in st.session_state
    if has_imputed:
        # print("Imputed file exists:", st.session_state["imputed_file"])
        pass

    if column_name not in data.columns:
        st.error(f"Column '{column_name}' not found in the dataset.")
        return

    column_data = data[column_name]
    valid_data = column_data.dropna()
    nan_count = column_data.isna().sum()

    # Ensure valid_data contains only numeric values
    valid_data = pd.to_numeric(valid_data, errors='coerce').dropna()

    # Check if valid_data is empty
    if valid_data.empty:
        st.error("No valid numeric data available for the selected column.")
        return

    # Create histogram bins from valid data
    hist_series, bins = pd.cut(valid_data, bins=20, retbins=True)
    hist_values = hist_series.value_counts(sort=False)

    # Convert bins to string labels for readability
    bin_labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

    # Initialize bar counts
    imputed_hist_values = [0] * len(bin_labels)

    # If imputed_file exists, calculate histogram for imputed values
    if has_imputed:
        imputed_data = st.session_state["imputed_file"]
        if column_name in imputed_data.columns:
            original_nan_mask = column_data.isna()
            imputed_values = imputed_data.loc[original_nan_mask, column_name]
            imputed_bins = pd.cut(imputed_values, bins=bins)
            imputed_counts = imputed_bins.value_counts(sort=False)

            # Align imputed bin counts with all bin intervals (even if 0)
            imputed_hist_values = [imputed_counts.get(interval, 0) for interval in pd.IntervalIndex.from_breaks(bins)]


    # Create a Plotly figure
    fig = go.Figure()

    # Add original data histogram bars (blue)
    fig.add_trace(go.Bar(x=bin_labels, y=hist_values, marker_color='blue', name='Data Distribution'))

    if not has_imputed:
        fig.add_trace(go.Bar(x=['NaN Values'], y=[nan_count], marker_color='orange', name='NaN Count'))

    # Add imputed data histogram bars (yellow), overlaid
    if has_imputed:
        if column_name in imputed_data.columns:
            imputed_nan_count = imputed_data[column_name].isna().sum()
            if imputed_nan_count > 0:
                fig.add_trace(go.Bar(x=['Empty in Imputed'], y=[nan_count], marker_color='red', name='Empty in Imputed'))
            else:        
                # Add imputed data histogram bars (yellow)
                fig.add_trace(go.Bar(x=bin_labels, y=imputed_hist_values, marker_color='yellow', name='Imputed Values'))

    # Update layout
    fig.update_layout(
        xaxis_title='Value Ranges',
        yaxis_title='Frequency',
        barmode='stack'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)