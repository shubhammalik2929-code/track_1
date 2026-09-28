# 🛒 Automated Shelf Auditing & Real-Time Stockout Detection

> **E-Cell IIT Bombay Computer Vision AI Hackathon**  
> *Powered by Everseen Architecture*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-yellow)

---

## 📌 Overview

An edge-accelerated computer vision and automated shelf auditing platform designed to eliminate retail out-of-stock downtime and enforce planogram compliance in real time. 

Using retail CCTV camera streams, the system maps physical shelves to a digital twin using homography calibration, detects SKU voids, filters transient shopper interactions using temporal persistence tracking, and routes actionable restock alerts to store staff.

---

## ✨ Key Features

- **📹 Live CCTV Aisle Feed & CV Inference:** Multi-camera selector with synthetic retail shelf simulation, bounding boxes, and camera OSD telemetry.
- **⚡ Real-Time Alert Operations Center:** Interactive stockout alert assignment with staff dispatch workflows (`st.session_state`).
- **📐 Planogram Digital Twin:** Slot-by-slot visual matrix comparing planned product facings against physical detections.
- **📊 Store Analytics & Revenue ROI:** Hourly stockout occurrence charts, category vulnerability heatmaps, and staff restock SLA leaderboards.
- **🚨 Incident Log & Export:** Complete searchable historical anomaly log with one-click CSV export for store managers.
- **📤 Custom Inference Testing:** Upload custom shelf images or test videos to test homography and bounding box detection.

---

## 🏗️ System Architecture
