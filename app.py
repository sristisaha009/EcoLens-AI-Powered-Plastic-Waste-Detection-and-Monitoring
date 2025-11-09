import streamlit as st
from streamlit_folium import st_folium
import folium
import os, json
import pandas as pd
import matplotlib.pyplot as plt

from utils.image_utils import save_uploaded_file, load_image_cv2, pil_from_cv2, try_extract_exif_gps, OUTPUT_DIR
from utils.detection_utils import load_model, run_yolo_inference, generate_heatmap_from_detections, overlay_heatmap, compute_severity_score
from utils.location_utils import get_geolocation, inject_geolocation_js, get_ip_geolocation
import db

# ✅ Updated Title and Page Config
st.set_page_config(layout="wide", page_title="🌱 EcoLens — AI-Powered Plastic Waste Detection")

# App Title
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌱 EcoLens: AI-Powered Plastic Waste Detection and Monitoring</h1>", unsafe_allow_html=True)
st.markdown("---")

# init DB
db.init_db()

# sidebar
st.sidebar.title("Options")
model_path = st.sidebar.text_input("Model path", value="best.pt")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
sigma = st.sidebar.slider("Heatmap sigma (px)", 10, 200, 50, 10)
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Report (Citizen)", "NGO Dashboard"])

# load model once
try:
    model = load_model(model_path)
except Exception as e:
    st.sidebar.error(f"Model load error: {e}")
    model = None

# ============================ CITIZEN MODE ============================
if mode == "Report (Citizen)":
    st.header("📷 Report Plastic Waste")
    uploaded = st.file_uploader("Upload an image of river/beach", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    notes = st.text_area("Notes (optional)")

    # --- Auto location ---
    lat, lon = get_geolocation()
    if not lat:
        inject_geolocation_js()
    if not lat:
        lat, lon = get_ip_geolocation()

    if lat and lon:
        st.success(f"📍 Location detected automatically: {lat:.5f}, {lon:.5f}")
    else:
        st.warning("⚠️ Could not fetch location. Will try EXIF metadata from image.")

    # --- Initialize session state for results ---
    if "citizen_results" not in st.session_state:
        st.session_state["citizen_results"] = None

    if st.button("Submit report"):
        if uploaded is None:
            st.error("Please upload an image")
        elif model is None:
            st.error("Model not loaded")
        else:
            saved_path = save_uploaded_file(uploaded)

            # try EXIF GPS if browser/IP failed
            if not lat or not lon:
                coords = try_extract_exif_gps(saved_path)
                if coords:
                    lat, lon = coords

            # run inference
            img = load_image_cv2(saved_path)
            dets, ann = run_yolo_inference(model, img, conf=conf_thresh)
            heatmap = generate_heatmap_from_detections(img.shape, dets, sigma=sigma)
            overlay = overlay_heatmap(img, heatmap, alpha=0.45)
            sev = compute_severity_score(dets)

            # save annotated & overlay
            base = os.path.splitext(os.path.basename(saved_path))[0]
            ann_path = os.path.join(OUTPUT_DIR, f"{base}_annotated.png")
            overlay_path = os.path.join(OUTPUT_DIR, f"{base}_overlay.png")
            pil_from_cv2(ann).save(ann_path)
            pil_from_cv2(overlay).save(overlay_path)

            # categories summary
            cats = {}
            for d in dets:
                cats[d["class_name"]] = cats.get(d["class_name"], 0) + 1
            cats_json = json.dumps(cats)

            # insert into DB
            rec_id = db.add_report(os.path.basename(saved_path), lat, lon, cats_json, float(sev), notes)
            st.success(f"Report submitted (id={rec_id}). Severity: {sev:.2f}")

            # store results in session state for toggling later
            st.session_state["citizen_results"] = {
                "overlay": overlay,
                "ann": ann,
                "cats": cats
            }

    # --- Show results if available ---
    if st.session_state["citizen_results"] is not None:
        res = st.session_state["citizen_results"]
        vis_type = st.radio("Choose visualization", ["Heatmap Overlay", "Predicted Bounding Boxes"])
        if vis_type == "Heatmap Overlay":
            st.image(pil_from_cv2(res["overlay"]), caption="Heatmap overlay", use_container_width=True)
        else:
            st.image(pil_from_cv2(res["ann"]), caption="Predicted Bounding Boxes", use_container_width=True)

        st.write("Detected categories:", res["cats"])

# ============================ NGO DASHBOARD ============================
elif mode == "NGO Dashboard":
    if "ngo_authenticated" not in st.session_state:
        st.session_state["ngo_authenticated"] = False

    if not st.session_state["ngo_authenticated"]:
        password_input = st.text_input("Enter NGO password to access dashboard", type="password")
        if st.button("Submit"):
            if password_input == "ngo2025":
                st.session_state["ngo_authenticated"] = True
                st.success("✅ Password correct! Access granted.")
            else:
                st.error("❌ Incorrect password. Try again.")

    if st.session_state["ngo_authenticated"]:
        st.header("📊 NGO Dashboard")
        st.markdown("Visualize crowdsourced reports, heatmaps, and trends.")
        reports = db.get_all_reports()
        if not reports:
            st.info("No reports yet — ask citizens to submit images!")
        else:
            df = pd.DataFrame(reports)

            # --- Category Filter ---
            all_categories = set()
            for r in reports:
                cats = json.loads(r["categories"])
                all_categories.update(cats.keys())
            selected_categories = st.multiselect("Filter reports by category", options=sorted(all_categories),
                                                 default=list(all_categories))

            if selected_categories:
                df = df[df["categories"].apply(lambda c: any(cat in json.loads(c) for cat in selected_categories))]

            # left: map
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🗺️ Map of Reports")
                avg_lat = df["lat"].dropna().mean() if not df["lat"].dropna().empty else 0
                avg_lon = df["lon"].dropna().mean() if not df["lon"].dropna().empty else 0
                m = folium.Map(location=[avg_lat or 0, avg_lon or 0], zoom_start=4)
                for r in df.to_dict(orient="records"):
                    lat, lon = r["lat"], r["lon"]
                    popup = folium.Popup(
                        html=f"ID: {r['id']}<br>Severity: {r['severity']:.2f}<br>Categories: {r['categories']}",
                        max_width=300)
                    if lat and lon:
                        folium.CircleMarker(location=[lat, lon], radius=7, color="red" if r["severity"] > 1 else "blue",
                                            fill=True, popup=popup).add_to(m)
                st_folium(m, width=700, height=450)

            with col2:
                st.subheader("📈 Summary & Charts")
                summary = db.get_summary(selected_categories)
                st.metric("Total reports", summary["total_reports"])
                st.metric("Avg severity", f"{summary['avg_severity']:.2f}")

                cat_counts = summary["category_counts"]
                if cat_counts:
                    cat_df = pd.DataFrame(list(cat_counts.items()), columns=["category", "count"]).sort_values("count",
                                                                                                               ascending=False)
                    st.bar_chart(cat_df.set_index("category"))

                st.subheader("Recent Reports")
                st.dataframe(df[["id", "filename", "timestamp", "lat", "lon", "severity"]].head(20))
                sel = st.selectbox("View report id", options=df["id"].tolist())
                rec = next((r for r in df.to_dict(orient="records") if r["id"] == sel), None)
                if rec:
                    st.write("Report details")
                    st.write(rec)

                    # --- Visualization toggle for NGO ---
                    vis_type = st.radio(f"Visualization for report {rec['id']}",
                                        ["Heatmap Overlay", "Predicted Bounding Boxes"])
                    basename = os.path.splitext(rec["filename"])[0]
                    ann_path = os.path.join(OUTPUT_DIR, f"{basename}_annotated.png")
                    overlay_path = os.path.join(OUTPUT_DIR, f"{basename}_overlay.png")
                    if vis_type == "Heatmap Overlay" and os.path.exists(overlay_path):
                        st.image(overlay_path, caption="Heatmap Overlay", use_container_width=True)
                    elif vis_type == "Predicted Bounding Boxes" and os.path.exists(ann_path):
                        st.image(ann_path, caption="Predicted Bounding Boxes", use_container_width=True)
