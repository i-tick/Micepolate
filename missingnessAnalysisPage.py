import random
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from customLableEncoder import *
from columnHistogram import load_histogram
from missingnessScatter import * 
from logisticMissingnessSignificance import *
from stackedCategoricalCounts import *
from mice import *
import numpy as np

def highlight_nan_df(file):
    def highlight_index(index):
        if index.name in ["Datatype", "Nulls"]:
            return ['background-color: #b3a6ed'] * len(file.columns)
        else:
            return [''] * len(file.columns)

    styled_df = file.style.apply(highlight_index, axis=1)
    styled_df = styled_df.highlight_null(color='#f2c0b3')
    return styled_df


def update_data_types(new_data_type_col, new_data_type, treat_none_as_category, custom_encoder=False):
    if new_data_type == "Categorical":
        le = CustomLabelEncoder(treat_none_as_category = treat_none_as_category) if custom_encoder else LabelEncoder()

        # Only fill NaNs if not using CustomLabelEncoder
        if not custom_encoder:
            st.session_state.file[new_data_type_col].fillna('SPECIFICALLY_MARKED_MISSING_CATEGORY_PREPROCESSING_DATA', inplace=True)
        
        st.session_state.file[new_data_type_col] = st.session_state.file[new_data_type_col].astype("str", errors="ignore")

        if custom_encoder:
            # Transform with CustomLabelEncoder without replacing NaNs
            st.session_state.file[new_data_type_col] = le.fit_transform(st.session_state.file, new_data_type_col)
        else:
            # Transform with basic LabelEncoder and replace the missing marker with NaN
            st.session_state.file[new_data_type_col] = le.fit_transform(st.session_state.file[new_data_type_col])
            if "SPECIFICALLY_MARKED_MISSING_CATEGORY_PREPROCESSING_DATA" in le.classes_:
                missing_val = le.transform(["SPECIFICALLY_MARKED_MISSING_CATEGORY_PREPROCESSING_DATA"])[0]
                st.session_state.file[new_data_type_col].replace(missing_val, np.nan, inplace=True)

        st.session_state["label_encoder_" + str(new_data_type_col)] = le
        # print(st.session_state["label_encoder_" + str(new_data_type_col)].category_to_mean_map if custom_encoder else le.classes_)

    else:
        st.session_state.file[new_data_type_col] = st.session_state.file[new_data_type_col].astype(new_data_type, errors="ignore")
    st.rerun()

def restructure_dataframe(data):
    local_df = pd.DataFrame(data)
    cols = []
    datatype = []
    nulls = []
    for i in range(len(local_df.columns)):
        cols.append(local_df.columns[i])
        datatype.append(str(local_df.dtypes.loc[local_df.columns[i]]))
        nulls.append(str(int(local_df.isnull().sum()[local_df.columns[i]])))

    # Create DataFrame
    info_df = pd.DataFrame([datatype, nulls], columns=cols, index=['Datatype', 'Nulls'])

    # print(info_df)
    combined_df = pd.concat([info_df, local_df], axis=0)

    return combined_df

def render_missingness():
    # Add custom CSS for better spacing and reduced size
    st.markdown(
        """
        <style>
        .change-data-types {
            padding-left: 20px; /* Add spacing to the left */
            border-left: 2px solid #eaeaea; /* Add a vertical separator */
            font-size: 0.9rem; /* Reduce font size */
            max-width: 300px; /* Restrict maximum width */
        }
        .change-data-types select, .change-data-types button {
            font-size: 0.8rem; /* Reduce font size for dropdowns and buttons */
            padding: 5px 10px; /* Adjust padding */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # If a file is uploaded, show the preprocessed data
    if st.session_state.get("file_uploaded", False):
        styled_df = highlight_nan_df(restructure_dataframe(st.session_state.file))
        df = st.session_state.file
        # print(df)
        target_df = st.session_state.file
        st.session_state["NaN-mask"] = target_df.isna()

        # Create two columns for layout
        col1, col2, col3 = st.columns([2, 2, 3], gap="medium")  # Adjust column widths (reduce right column)

        # Display the uploaded data on the left
        with col1:
            # st.markdown('<div class="uploaded-data">', unsafe_allow_html=True)
            # st.write("Uploaded Data:")
            # st.write(styled_df)
            # st.markdown('</div>', unsafe_allow_html=True)
            # Display the dataframe
            
            # Display interactive table with proper selection mode
            imputed_df = st.session_state.get("imputed_file", df)
            nan_mask = df.isna()

            # Define a styling function
            def highlight_imputed(val, is_nan):
                return 'background-color: #ff0000' if is_nan else ''

            # Create a styled DataFrame using apply with the mask
            styled_df = imputed_df.style.apply(
                lambda col: [highlight_imputed(val, nan) for val, nan in zip(col, nan_mask[col.name])],
            )
            selection = st.dataframe(
                styled_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["multi-column"],  # Use the correct format
                hide_index=False,
                height=600  # Increase the height of the dataframe
            )


            # Initialize previous_selections in session state if it doesn't exist
            if "previous_selections" not in st.session_state:
                st.session_state.previous_selections = []

            # Check if selection exists and has the proper structure
            if selection and hasattr(selection, "selection") and hasattr(selection.selection, "columns"):
                # If columns were selected, update the session state
                if len(selection.selection.columns) > 0:
                    st.session_state.previous_selections = selection.selection.columns
                selected_columns = st.session_state.previous_selections
                # st.write(f"Selected columns: {selected_columns}")
            else:
                # Use previously selected columns
                selected_columns = st.session_state.previous_selections
                st.write(f"Using previously selected columns: {selected_columns}")

            

        with col2:
            st.markdown('<h5 style="text-align: center;">Change Data Types </h5>', unsafe_allow_html=True)

            u_col1, u_col2 = st.columns([2, 2], gap="small")

            with u_col1:
                new_data_type_col = st.selectbox("Select the column: ", list(df.columns), index=list(df.columns).index(selected_columns[0]) if selected_columns else 0)
            with u_col2:
                new_data_type = st.selectbox("Select the Data Type: ", ["int64", "float64", "string", "bool", "datetime64[ns]", "Categorical"])

            if new_data_type == "Categorical":
                treat_none_as_category = st.checkbox("Treat None as a Category", value=False)

                if st.button("Update with Max Corr Label Encoder", key="update_data_type_button_custom"):
                    if new_data_type_col != "Select Column" and new_data_type != "Select Data Type":
                        update_data_types(new_data_type_col, new_data_type, treat_none_as_category= treat_none_as_category, custom_encoder=True)

            else:
                if st.button("Update Data Type", key="update_data_type_button"):
                    if new_data_type_col != "Select Column" and new_data_type != "Select Data Type":
                        update_data_types(new_data_type_col, new_data_type)

            configure_mice(selected_columns)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.write("#### Histogram Visualization")
            columns = st.session_state.file.columns.tolist()
            
            # preserve the previous selected column in session state
            selected_column = st.selectbox(
                "Select a column for the histogram",
                columns,
                index=columns.index(st.session_state.get("previous_histogram_column", columns[0]))
            )

            # Save the selected column in session state
            st.session_state["previous_histogram_column"] = selected_column

            if selected_column:
                load_histogram(st.session_state.file, selected_column)


        col1, col2, col3 = st.columns([3, 2, 2], gap="medium")
        with col1:
            render_missingness_scatter_plot()
        with col2:
            populateMissingnessSignificanceHeatmap()
        with col3:
            plot_stacked_bar_graph()

def load_file():
    if "file_uploaded" not in st.session_state or not st.session_state.file_uploaded:
        file = st.file_uploader("Upload File Here", type={"csv", "tsv", "xlsx"},accept_multiple_files=False)
        if file is not None:
            if str(file.name).endswith(".csv"):
                st.session_state.file = pd.read_csv(file)
            elif str(file.name).endswith(".tsv"):
                st.session_state.file = pd.read_csv(file, sep="\t")
            elif str(file.name).endswith(".xlsx"):
                st.session_state.file = pd.read_excel(file)

            if not st.session_state.file.empty:
                st.session_state.file_uploaded = True
                render_missingness()
                st.rerun()
            else:
                st.error("File format not accepted or the file is empty.")
    else:
        render_missingness()

def configure_mice(selected_columns):
    df = st.session_state.get("file", None)

    if df is not None:
        df = st.session_state.get("file", None)
        st.markdown("<h5 style='text-align: center;'>Configure MICE</h5>", unsafe_allow_html=True)
        selected_columns_for_mice = st.multiselect("Select the columns you want to impute:", st.session_state.file.columns,
                                                   key="columns_for_mice", default=selected_columns)
        k = 0
        error_message = ""

        if selected_columns_for_mice:
            # if len(selected_columns_for_mice) == 1:
            #     error_message = "You need more than one column to run the MICE Algorithm"
            #     k = 1
            for column in selected_columns_for_mice:
                if st.session_state.file[column].dtype.kind not in "buifc":
                    error_message += f"The column '{column}' isn't Numeric or Boolean and hence can't be considered for MICE Algorithm\n"
                    k = 1
        if selected_columns_for_mice is None or len(selected_columns_for_mice) == 0:
            k = 1

        max_iterations = st.number_input("Max Iterations", min_value=1, max_value=1000, value=25, step=1)

        random_state = random.randint(0, 9999)

        if k == 1:
            if error_message != '':
                st.error(error_message)
            col1, col2 = st.columns(2)
            with col1:
                st.button("Run MICE", key="run_mice_button", disabled=True)
            with col2:
                st.button("RUN BART", key="reset_mice_button", disabled=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run MICE", key="run_mice_button", use_container_width=True):
                    new_file, _ = implement_MICE(df, selected_columns_for_mice, max_iterations, random_state)
                    st.session_state["imputed_file"] = new_file
                    st.rerun()
            with col2:
                if st.button("RUN BART", key="reset_mice_button", use_container_width=True):
                    new_file, _ = implement_BART(df, selected_columns_for_mice, random_state)
                    st.session_state["imputed_file_bart"] = new_file
                    st.rerun()

        if st.session_state.get("imputed_file", None) is not None:
            if st.button("Go to Analysis Page", key="go_to_analysis_page"):
                st.session_state.page = 2
                st.rerun()
    else:
        st.error("No data available in 'st.session_state['file']'.")
