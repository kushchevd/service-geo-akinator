import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium


def draw_map(lat=None, lon=None, markers=None, color="red"):
    if markers is None:
        markers = []

    if lat is None and lon is None and not markers:
        map = folium.Map(location=[0, 0], zoom_start=1)
    elif lat is not None and lon is not None:
        markers.append({"coords": [lat, lon], "color": color})
        map = folium.Map(location=markers[-1]["coords"], zoom_start=3)
    elif markers:
        map = folium.Map(location=markers[-1]["coords"], zoom_start=3)
    else:
        st.error("Set both lat and lon")
        map = folium.Map(location=[0, 0], zoom_start=1)
    for marker in markers:
        folium.Marker(
            location=marker["coords"],
            popup=f"Latitude: {marker['coords'][0]:.4f}, Longitude: {marker['coords'][1]:.4f}",
            icon=folium.Icon(color=marker["color"]),
        ).add_to(map)
    map_data = st_folium(
        map,
        width=800,
        height=500,
        returned_objects=["last_clicked"],
        key="correction" if st.session_state.correction_mode else "prediction",
    )

    return map_data, markers


if "correction_mode" not in st.session_state:
    st.session_state.correction_mode = False
if "predicted" not in st.session_state:
    st.session_state.predicted = False

st.title("🌍 City Location Predictor")

prompt = st.text_area("Enter a description of a city or place:", height=150)

if st.button("Predict Location") and prompt:
    (
        st.session_state.correction_mode,
        st.session_state.lat_cor,
        st.session_state.lon_cor,
    ) = (False, None, None)
    try:
        response = requests.post(
            f"{os.getenv('BACKEND_URL')}/forward", json={"text": prompt}, timeout=10
        )
        if response.status_code == 200:
            data = response.json()["predictions"]
            st.session_state.last_lat_pred, st.session_state.last_lon_pred = (
                data["lat"],
                data["lon"],
            )
            st.success(
                f"🌐 Predicted Coordinates: Latitude = {st.session_state.last_lat_pred:.4f}°,"
                "Longitude = {st.session_state.last_lon_pred:.4f}°"
            )
            _, st.session_state.markers = draw_map(
                lat=st.session_state.last_lat_pred,
                lon=st.session_state.last_lon_pred,
            )
            st.session_state.predicted = True
            st.rerun()
        else:
            st.error("Error to get coordinates")
    except Exception as e:
        st.error(f"Unpredictable error: {e}")


if st.session_state.predicted:
    if not st.session_state.correction_mode:
        draw_map(
            lat=st.session_state.last_lat_pred,
            lon=st.session_state.last_lon_pred,
        )
        if st.button("Model is wrong!"):
            st.session_state.correction_mode = True
            st.rerun()

if st.session_state.correction_mode:
    st.write("Plese set a dot on the correct place")
    map_data, _ = draw_map(
        markers=st.session_state.markers,
        color="green",
    )
    if map_data.get("last_clicked") is not None:
        st.session_state.markers = [
            m for m in st.session_state.markers if m["color"] != "green"
        ]
        st.session_state.markers.append(
            {
                "coords": [
                    map_data.get("last_clicked")["lat"],
                    map_data.get("last_clicked")["lng"],
                ],
                "color": "green",
            }
        )
        st.rerun()

    if st.button("Submit correct place"):
        add_data_response = requests.put(
            f"{os.getenv('BACKEND_URL')}/add_data",
            json={
                "text": prompt,
                "coords": {
                    "lat": st.session_state.markers[-1]["coords"][0],
                    "lon": st.session_state.markers[-1]["coords"][1],
                },
            },
            timeout=10,
        )
        if add_data_response.status_code == 200:
            st.success("Changes saved")
        else:
            st.error("Unexpected error, please try again")
