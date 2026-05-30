
import streamlit as st
import pandas as pd

from utils.timetable import (
    get_all_stations
)

from utils.eta import (
    generate_eta_table
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SmartTransit Hyderabad",
    page_icon="🚆",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚆 SmartTransit Hyderabad")

st.markdown(
    """
    Predict MMTS train arrival times using:
    - timetable intelligence
    - route traversal
    - ML-powered delay estimation
    """
)

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Journey Input")

stations = get_all_stations()

selected_station = st.sidebar.selectbox(
    "Select Departure Station",
    stations
)

selected_time = st.sidebar.text_input(
    "Current Time (HH:MM:SS)",
    "18:10:00"
)

find_train = st.sidebar.button(
    "Find Next Train"
)

# =========================================================
# MAIN APP LOGIC
# =========================================================

if find_train:

    with st.spinner("Finding next train..."):

        results = generate_eta_table(
            selected_station,
            selected_time
        )

    # -----------------------------------------------------
    # NO TRAIN FOUND
    # -----------------------------------------------------

    if results is None:

        st.error(
            "No upcoming trains found."
        )

    else:

        # -------------------------------------------------
        # TRAIN INFO
        # -------------------------------------------------

        st.subheader("Next Available Train")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Train ID",
                value=results["train_id"]
            )

        with col2:

            st.metric(
                label="Predicted Delay",
                value=f'{results["predicted_delay"]:.2f} min'
            )

        # -------------------------------------------------
        # ETA TABLE
        # -------------------------------------------------

        st.subheader("Predicted Arrival Times")

        eta_table = results["eta_table"].copy()

        # format datetime columns nicely
        eta_table["arrival_time"] = (
            eta_table["arrival_time"]
            .dt.strftime("%H:%M:%S")
        )

        eta_table["departure_time"] = (
            eta_table["departure_time"]
            .dt.strftime("%H:%M:%S")
        )

        eta_table["predicted_arrival"] = (
            eta_table["predicted_arrival"]
            .dt.strftime("%H:%M:%S")
        )

        eta_table["predicted_departure"] = (
            eta_table["predicted_departure"]
            .dt.strftime("%H:%M:%S")
        )

        st.dataframe(
            eta_table,
            use_container_width=True
        )

        # -------------------------------------------------
        # DELAY INSIGHT
        # -------------------------------------------------

        if results["predicted_delay"] > 10:

            st.warning(
                "High operational delay expected."
            )

        elif results["predicted_delay"] > 5:

            st.info(
                "Moderate delays expected."
            )

        else:

            st.success(
                "Train expected to run near schedule."
            )

# =========================================================
# DEFAULT LANDING PAGE
# =========================================================

else:

    st.markdown(
        """
        ## How To Use

        1. Select your departure station  
        2. Enter the current time  
        3. Click **Find Next Train**  
        4. View predicted ETAs for downstream stations

        ---

        This project combines:
        - GTFS timetable data
        - Route traversal logic
        - Machine learning delay estimation
        """
    )
