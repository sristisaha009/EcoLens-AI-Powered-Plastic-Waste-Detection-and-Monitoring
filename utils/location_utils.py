import streamlit as st
import requests

def get_geolocation():
    """Fetch user location from query params (set via JS)."""
    location = st.query_params.get("location", None)
    if location:
        # query_params values are lists
        if isinstance(location, list):
            location = location[0]
        lat, lon = map(float, location.split(","))
        return lat, lon
    return None, None

def inject_geolocation_js():
    """Inject JS to request browser geolocation and reload page with ?location=lat,lon"""
    st.markdown("""
        <script>
        navigator.geolocation.getCurrentPosition(function(position) {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            const newUrl = window.location.protocol + "//" + window.location.host + window.location.pathname + "?location=" + lat + "," + lon;
            window.location.href = newUrl;
        });
        </script>
    """, unsafe_allow_html=True)

def get_ip_geolocation():
    """Fallback: Get location from IP (less accurate)."""
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        lat, lon = map(float, data["loc"].split(","))
        return lat, lon
    except:
        return None, None
