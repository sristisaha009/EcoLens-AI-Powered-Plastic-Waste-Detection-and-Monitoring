# 🌱 EcoLens: AI-Powered Plastic Waste Detection and Monitoring

This is an AI-driven platform designed to detect and map plastic waste in rivers and beaches. It combines computer vision, geolocation, and data visualization to empower citizens and NGOs to collaboratively monitor and combat water pollution.

---

## 🧩 Overview

Plastic pollution severely threatens aquatic life and ecosystems. **EcoLens** enables citizens to upload images of polluted rivers or beaches, which are analyzed using a YOLO-based object detection model. The system generates heatmaps showing waste concentration and provides NGOs with an interactive dashboard for monitoring pollution severity and planning cleanups effectively.

---

## ⚙️ Key Features

- 🧠 **AI Detection:** YOLO-based deep learning model identifies and classifies plastic waste.  
- 🌍 **Automatic Geolocation:** Detects GPS coordinates using browser, IP, or EXIF data.  
- 🔥 **Heatmap Generation:** Visualizes pollution density and calculates severity scores.  
- 👥 **Citizen Reporting:** Easy image upload and waste reporting for everyone.  
- 🗺️ **NGO Dashboard:** Interactive map, filters, and category analytics for organizations.  
- 🔐 **Secure Access:** NGO dashboard protected by a password system.  
- 🔄 **Visualization Toggle:** Switch between bounding box view and heatmap overlay.

---

## 🧠 Tech Stack

### Frontend
- Streamlit  
- Streamlit-Folium / Folium  
- Matplotlib  
- Pandas  

### Backend
- Python  
- Ultralytics YOLOv8  
- OpenCV  
- NumPy  

### Database
- SQLite (local) / MySQL (scalable)  

### Geolocation
- HTML5 Geolocation API  
- IP-based location detection  
- EXIF GPS extraction (via Pillow)

---

## 🏗️ System Architecture

Citizen Upload → YOLO Detection → Heatmap + Severity → Geolocation Extraction → Database → NGO Dashboard

---


## 📊 Dataset

* **Google Images**

---

## 🧭 Future Enhancements

* Real-time waste tracking with satellite/Aerial data
* Mobile app for on-site waste reporting
* AI-based trend prediction for pollution patterns
* Multi-lingual support for global citizen engagement

---

## 💡 Impact

EcoLens supports **UN Sustainable Development Goal 14 – Life Below Water**, promoting clean oceans and sustainable ecosystems through AI-powered monitoring and community participation.
