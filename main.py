import streamlit as st
from analysisPage import *
from missingnessAnalysisPage import load_file, second_page_content


st.set_page_config(layout='wide')
pd.set_option("styler.render.max_elements", 14758254)

def next_page():
    st.session_state.page += 1

def home_page_content():
    st.session_state.page = 0
    st.rerun()

def new_data_page():
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]
    st.rerun()

if "page" not in st.session_state:
    st.session_state.page = 0

def get_page():
    if st.session_state.page == 0:
        load_file()
    elif st.session_state.page == 1:
        second_page_content()
    elif st.session_state.page == 2:
        analysis_page_content()

def header():
    file_uploaded = st.session_state.get('file', None) is not None
    st.markdown(
        """
        <style>
        /* Remove excess top padding */
        .block-container {
            padding-top: 3rem;
        }
        /* Adjust buttons to ensure proper sizing and alignment */
        .stButton button {
            width: 100%;
            height: 2.5rem;
            margin-top: 0;
            margin-bottom: 0;
        }
        /* Ensure header text aligns properly with buttons */
        h1 {
            line-height: 2.5rem;
            margin: 0;
            padding-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 6, 1])

    with left:
        if st.button(":classical_building: Home", key="home_button"):
            home_page_content()

    with center:
        if file_uploaded:
            st.markdown(
                "<h2 style='text-align: center; color: MediumSeaGreen;'>Missing Data Estimator</h2>",
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                "<h1 style='text-align: center; color: MediumSeaGreen;'>Missing Data Estimator</h1>",
                unsafe_allow_html=True,
            )

    with right:
        if st.button(":repeat: New Data", key="new_data_button"):
            new_data_page()


if __name__ == "__main__":
    header()
    get_page()