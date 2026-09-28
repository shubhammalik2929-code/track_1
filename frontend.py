"""
================================================================================
Everseen x IIT Bombay - Retail AI Operations Hub (Frontend)
File: frontend.py
Author: AI/ML & Computer Vision Lead
Architecture: Streamlit Cyber-Retail NOC + Dual RBAC Dashboards + Quad CCTV
================================================================================
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
import base64
import io
from PIL import Image

# ==============================================================================
# 1. PAGE CONFIGURATION & REFINED CYBER-RETAIL NOC THEME
# ==============================================================================
st.set_page_config(
    page_title="Everseen x IIT Bombay - Retail AI Operations Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_API_BASE = "http://127.0.0.1:8000"

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #050811;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* ---------------------------------------------------------
       MATCHED ELECTRIC NEON AMBIENT BACKGROUND (LOGIN SCREEN)
       --------------------------------------------------------- */
    .iitb-bg-ambient {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .ambient-orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(110px);
        opacity: 0.38;
        animation: floatOrb 10s ease-in-out infinite alternate;
    }
    .orb-1 {
        width: 580px;
        height: 580px;
        background: radial-gradient(circle, #00f0ff 0%, rgba(0, 240, 255, 0.12) 50%, transparent 70%);
        top: 6%;
        left: 16%;
    }
    .orb-2 {
        width: 620px;
        height: 620px;
        background: radial-gradient(circle, #2563eb 0%, rgba(37, 99, 235, 0.15) 50%, transparent 70%);
        bottom: 4%;
        right: 16%;
        animation-delay: -5s;
    }
    @keyframes floatOrb {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(45px, -30px) scale(1.15); }
        100% { transform: translate(-35px, 35px) scale(0.92); }
    }

    /* Giant 3D Popping Typography */
    .iitb-popping-watermark {
        position: absolute;
        top: 38%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: clamp(65px, 12vw, 155px);
        font-weight: 900;
        letter-spacing: 18px;
        white-space: nowrap;
        user-select: none;
        text-transform: uppercase;
        color: #e0f2fe;
        text-shadow: 
            0 0 10px rgba(0, 240, 255, 0.9),
            0 0 25px rgba(0, 240, 255, 0.7),
            0 0 50px rgba(14, 165, 233, 0.55),
            0 0 90px rgba(37, 99, 235, 0.4),
            0 0 140px rgba(29, 78, 216, 0.25);
        animation: neonBreathing 5s ease-in-out infinite alternate;
    }
    .iitb-sub-watermark {
        position: absolute;
        top: 53%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: clamp(14px, 2.3vw, 24px);
        font-weight: 800;
        letter-spacing: 12px;
        color: #7dd3fc;
        text-shadow: 
            0 0 8px rgba(0, 240, 255, 0.7),
            0 0 20px rgba(37, 99, 235, 0.4);
        white-space: nowrap;
        user-select: none;
        text-transform: uppercase;
        animation: neonBreathing 5s ease-in-out infinite alternate;
        animation-delay: -2.5s;
    }
    @keyframes neonBreathing {
        0% {
            transform: translate(-50%, -50%) scale(0.96) perspective(600px) rotateX(4deg);
            opacity: 0.65;
            filter: brightness(0.95);
        }
        50% {
            opacity: 0.95;
            filter: brightness(1.2);
        }
        100% {
            transform: translate(-50%, -50%) scale(1.04) perspective(600px) rotateX(-3deg);
            opacity: 1;
            filter: brightness(1.35) drop-shadow(0 0 35px rgba(0, 240, 255, 0.5));
        }
    }

    /* Hackathon Pill */
    .hackathon-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(8, 14, 30, 0.85);
        border: 1px solid #00f0ff;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.35), inset 0 0 12px rgba(0, 240, 255, 0.15);
        color: #38bdf8;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 6px 20px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }

    /* Top Operations Bar */
    .noc-header {
        background: linear-gradient(90deg, #091024 0%, #0f1d3d 100%);
        border: 1px solid rgba(0, 240, 255, 0.3);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 240, 255, 0.1);
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .noc-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
    }
    .noc-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Telemetry Recording Badge */
    .rec-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #ef4444;
        border-radius: 50%;
        box-shadow: 0 0 10px #ef4444;
        animation: pulseAnimation 1.5s infinite;
    }
    @keyframes pulseAnimation {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.3; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Role Badges */
    .role-badge-owner {
        background: linear-gradient(135deg, #0284c7, #0369a1);
        color: #ffffff;
        font-weight: 800;
        font-size: 11px;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.5px;
        display: inline-block;
        border: 1px solid #38bdf8;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.5);
    }
    .role-badge-employee {
        background: linear-gradient(135deg, #0f766e, #115e59);
        color: #ffffff;
        font-weight: 800;
        font-size: 11px;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.5px;
        display: inline-block;
        border: 1px solid #2dd4bf;
        box-shadow: 0 0 14px rgba(45, 212, 191, 0.4);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(10, 17, 36, 0.88);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6), 0 0 25px rgba(0, 240, 255, 0.08);
    }
    .glass-card:hover {
        border-color: #00f0ff;
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.8), 0 0 35px rgba(0, 240, 255, 0.3);
        transform: translateY(-3px);
    }

    /* Stream Status Bar */
    .stream-status-bar {
        background-color: #080f24;
        color: #94a3b8;
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        display: flex;
        justify-content: space-between;
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-bottom: none;
    }

    /* Floor Plan Interactive SVG Card */
    .store-map-container {
        background: #080f24;
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }

    /* Timeline Stepper for Temporal Filter */
    .timeline-step {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .step-active {
        border-color: #00f0ff;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DEMO USERS DATABASE & AUTH SESSION
# ==============================================================================
USERS_DB = {
    "owner": {
        "password": "123",
        "role": "Shop Owner",
        "display_name": "Vikram Sethi (Store Director)",
        "email": "owner@everseen-retail.in",
        "badge_class": "role-badge-owner",
        "icon": "👑"
    },
    "rahul": {
        "password": "123",
        "role": "Floor Employee",
        "display_name": "Rahul Verma (Aisle 3 & 4 Lead)",
        "email": "rahul.v@everseen-retail.in",
        "badge_class": "role-badge-employee",
        "icon": "👷"
    },
    "priya": {
        "password": "123",
        "role": "Floor Employee",
        "display_name": "Priya Sharma (Aisle 1 & 2 Lead)",
        "email": "priya.s@everseen-retail.in",
        "badge_class": "role-badge-employee",
        "icon": "👷"
    }
}

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "temporal_test_sec" not in st.session_state:
    st.session_state.temporal_test_sec = 12


# ==============================================================================
# 3. STOCK INVENTORY DATABASE
# ==============================================================================
INVENTORY_STOCK_DB = [
    {"sku": "SKU #101", "name": "Coca-Cola 500ml", "category": "Beverages", "aisle": "Aisle 3", "shelf": "Shelf A (Slot 1)", "in_stock": 24, "min_threshold": 10, "unit_price": "₹45.00", "status": "In Stock", "barcode": "8901030381012"},
    {"sku": "SKU #102", "name": "Diet Coke 500ml", "category": "Beverages", "aisle": "Aisle 3", "shelf": "Shelf A (Slot 2)", "in_stock": 18, "min_threshold": 8, "unit_price": "₹50.00", "status": "In Stock", "barcode": "8901030381029"},
    {"sku": "SKU #103", "name": "Sprite 500ml", "category": "Beverages", "aisle": "Aisle 3", "shelf": "Shelf A (Slot 3)", "in_stock": 15, "min_threshold": 8, "unit_price": "₹45.00", "status": "In Stock", "barcode": "8901030381036"},
    {"sku": "SKU #201", "name": "Red Bull 250ml", "category": "Energy Drinks", "aisle": "Aisle 3", "shelf": "Shelf B (Slot 1)", "in_stock": 32, "min_threshold": 12, "unit_price": "₹125.00", "status": "In Stock", "barcode": "8901030382019"},
    {"sku": "SKU #202", "name": "Pepsi Zero 500ml", "category": "Beverages", "aisle": "Aisle 3", "shelf": "Shelf B (Slot 2)", "in_stock": 0, "min_threshold": 10, "unit_price": "₹45.00", "status": "OUT OF STOCK", "barcode": "8901030382026"},
    {"sku": "SKU #1102", "name": "Lays Cream & Onion 50g", "category": "Snacks", "aisle": "Aisle 1", "shelf": "Shelf A (Slot 2)", "in_stock": 6, "min_threshold": 15, "unit_price": "₹20.00", "status": "MISPLACED / LOW", "barcode": "8901030381102"},
    {"sku": "SKU #1105", "name": "Doritos Nacho Cheese 90g", "category": "Snacks", "aisle": "Aisle 1", "shelf": "Shelf A (Slot 3)", "in_stock": 28, "min_threshold": 10, "unit_price": "₹50.00", "status": "In Stock", "barcode": "8901030381105"},
    {"sku": "SKU #8820", "name": "Amul Butter 500g", "category": "Dairy", "aisle": "Aisle 2", "shelf": "Shelf C (Slot 1)", "in_stock": 42, "min_threshold": 15, "unit_price": "₹275.00", "status": "In Stock", "barcode": "8901030388820"},
    {"sku": "SKU #3312", "name": "Dove Moisturizing Soap 100g", "category": "Personal Care", "aisle": "Aisle 4", "shelf": "Shelf B (Slot 1)", "in_stock": 3, "min_threshold": 12, "unit_price": "₹65.00", "status": "CRITICAL LOW", "barcode": "8901030383312"},
    {"sku": "SKU #4589", "name": "Coca-Cola Zero Sugar 500ml", "category": "Beverages", "aisle": "Aisle 3", "shelf": "Shelf B (Slot 4)", "in_stock": 0, "min_threshold": 10, "unit_price": "₹45.00", "status": "OUT OF STOCK", "barcode": "8901030384589"},
]


# ==============================================================================
# 4. BACKEND API SERVICE BRIDGE WITH RESILIENT FALLBACK
# ==============================================================================
def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_API_BASE}/", timeout=1.0)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, {}

backend_online, backend_meta = check_backend_health()

def api_get_alerts():
    if backend_online:
        try:
            r = requests.get(f"{BACKEND_API_BASE}/api/v1/alerts", timeout=1.2)
            if r.status_code == 200:
                data = r.json()
                return data.get("active_alerts", []), data.get("resolved_alerts", [])
        except Exception:
            pass
    return st.session_state.get("fallback_active", []), st.session_state.get("fallback_resolved", [])

def api_resolve_alert(alert_id, staff_name):
    if backend_online:
        try:
            r = requests.post(
                f"{BACKEND_API_BASE}/api/v1/resolve-alert",
                json={"alert_id": alert_id, "resolved_by": staff_name},
                timeout=1.5
            )
            return r.status_code == 200
        except Exception:
            pass
    if "fallback_active" in st.session_state:
        for idx, a in enumerate(st.session_state.fallback_active):
            if a["alert_id"] == alert_id:
                res = st.session_state.fallback_active.pop(idx)
                res["status"] = "Completed"
                res["assigned_to"] = staff_name
                res["duration_sec"] = "Resolved"
                st.session_state.fallback_resolved.insert(0, res)
                return True
    return False

def api_simulate_alert(aisle, product, sku):
    if backend_online:
        try:
            r = requests.post(
                f"{BACKEND_API_BASE}/api/v1/alerts/simulate",
                json={"aisle": aisle, "product_name": product, "sku": sku, "category": "Beverages", "severity": "critical"},
                timeout=1.5
            )
            return r.status_code == 200
        except Exception:
            pass
    if "fallback_active" in st.session_state:
        st.session_state.fallback_active.insert(0, {
            "alert_id": f"ALT-{np.random.randint(200, 999)}",
            "aisle": aisle,
            "sku": sku,
            "product_name": product,
            "duration_sec": 8,
            "status": "Pending Restock",
            "assigned_to": "Unassigned",
            "severity": "critical"
        })
        return True
    return False

if "fallback_active" not in st.session_state:
    st.session_state.fallback_active = [
        {"alert_id": "ALT-101", "aisle": "Aisle 3, Shelf B", "sku": "SKU #4589", "product_name": "Coca-Cola 500ml", "duration_sec": 42, "status": "Pending Restock", "assigned_to": "Unassigned", "severity": "critical"},
        {"alert_id": "ALT-102", "aisle": "Aisle 1, Shelf A", "sku": "SKU #1102", "product_name": "Lays Cream & Onion", "duration_sec": 85, "status": "Pending Review", "assigned_to": "Unassigned", "severity": "warning"},
    ]
    st.session_state.fallback_resolved = [
        {"alert_id": "ALT-100", "aisle": "Aisle 2, Shelf C", "sku": "SKU #8820", "product_name": "Amul Butter 500g", "duration_sec": 130, "status": "Completed", "assigned_to": "Staff #4 (Ramesh)", "severity": "success"}
    ]


# ==============================================================================
# 5. SYNTHETIC CCTV CV STREAM GENERATOR
# ==============================================================================
def render_enhanced_shelf_frame(aisle_name, confidence_thresh=0.6, show_boxes=True, show_voids=True):
    frame = np.ones((440, 720, 3), dtype=np.uint8) * 35
    cv2.rectangle(frame, (20, 20), (700, 420), (16, 24, 48), -1)
    
    shelf_color = (30, 45, 80)
    cv2.rectangle(frame, (20, 135), (700, 150), shelf_color, -1)
    cv2.rectangle(frame, (20, 270), (700, 285), shelf_color, -1)
    cv2.rectangle(frame, (20, 400), (700, 415), (40, 55, 95), -1)

    cv2.putText(frame, "TIER A [HIGH MARGIN]", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1)
    cv2.putText(frame, "TIER B [EYE LEVEL]", (25, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1)
    cv2.putText(frame, "TIER C [BULK STOCK]", (25, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1)

    for i, x in enumerate([60, 165, 270, 375, 480, 585]):
        cv2.rectangle(frame, (x, 40), (x+75, 135), (35, 50, 85), -1)
        if show_boxes:
            cv2.rectangle(frame, (x, 40), (x+75, 135), (0, 240, 255), 2)
            cv2.putText(frame, f"SKU #10{i+1}", (x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 240, 255), 1)

    cv2.rectangle(frame, (60, 170), (145, 270), (35, 50, 85), -1)
    if show_boxes:
        cv2.rectangle(frame, (60, 170), (145, 270), (0, 240, 255), 2)
        cv2.putText(frame, "SKU #201", (60, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 240, 255), 1)

    if show_voids:
        cv2.rectangle(frame, (170, 170), (275, 270), (20, 20, 60), -1)
        cv2.rectangle(frame, (170, 170), (275, 270), (239, 68, 68), 2)
        cv2.line(frame, (170, 170), (275, 270), (239, 68, 68), 1)
        cv2.line(frame, (170, 270), (275, 170), (239, 68, 68), 1)
        cv2.rectangle(frame, (168, 150), (277, 168), (239, 68, 68), -1)
        cv2.putText(frame, "VOID: OUT OF STOCK", (172, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (255, 255, 255), 1)

    cv2.rectangle(frame, (300, 170), (385, 270), (45, 50, 75), -1)
    if show_boxes:
        cv2.rectangle(frame, (300, 170), (385, 270), (245, 158, 11), 2)
        cv2.putText(frame, "MISPLACED", (300, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (245, 158, 11), 1)

    for i, x in enumerate([410, 520, 615]):
        cv2.rectangle(frame, (x, 170), (x+75, 270), (35, 50, 85), -1)
        if show_boxes:
            cv2.rectangle(frame, (x, 170), (x+75, 270), (0, 240, 255), 2)
            cv2.putText(frame, f"SKU #20{i+4}", (x, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 240, 255), 1)

    for i, x in enumerate([60, 190, 320, 450, 580]):
        cv2.rectangle(frame, (x, 305), (x+100, 400), (35, 50, 85), -1)
        if show_boxes:
            cv2.rectangle(frame, (x, 305), (x+100, 400), (0, 240, 255), 2)
            cv2.putText(frame, f"SKU #30{i+1}", (x, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 240, 255), 1)

    cv2.rectangle(frame, (20, 20), (700, 46), (10, 17, 36), -1)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"FEED: {aisle_name.upper()} | {timestamp_str} | 30 FPS | YOLOv8-Retail",
                (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1)

    return frame


# ==============================================================================
# 6. AUTHENTICATION SCREEN WITH LIVE TELEMETRY & MATCHED NEON POPPING BACKGROUND
# ==============================================================================
if st.session_state.auth_user is None:
    st.markdown("""
        <div class="iitb-bg-ambient">
            <div class="ambient-orb orb-1"></div>
            <div class="ambient-orb orb-2"></div>
            <div class="iitb-popping-watermark">IIT BOMBAY</div>
            <div class="iitb-sub-watermark">E-CELL COMPUTER VISION HACKATHON 2026</div>
        </div>
    """, unsafe_allow_html=True)

    current_time_str = datetime.now().strftime("%H:%M:%S IST")
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 900px; margin: 0 auto; padding: 10px 20px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; border-bottom: 1px solid rgba(0, 240, 255, 0.2);">
            <div><span style="color: #00f0ff;">● NODE:</span> IITB-EDGE-MUM-01</div>
            <div><span style="color: #00f0ff;">● CV ENGINE:</span> YOLOv8x-Retail (TensorRT)</div>
            <div><span style="color: #00f0ff;">● LATENCY:</span> 16.4ms</div>
            <div><span style="color: #4ade80;">● CLOCK:</span> {current_time_str}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center; margin-top: 25px; margin-bottom: 25px; position: relative; z-index: 10;">
            <div class="hackathon-badge">🏛️ E-CELL IIT BOMBAY × EVERSEEN AI LABS</div>
            <h1 style="font-size: 34px; font-weight: 900; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 4px;">
                Retail Vision Operations Portal
            </h1>
            <div style="margin-top: 10px;">
                <div class="hackathon-badge" style="font-size: 12.5px; letter-spacing: 1.2px; padding: 7px 24px; margin-bottom: 0;">
                    👁️ Autonomous CCTV Shelf Auditing, Planogram Digital Twin & Dynamic Void Orchestration
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    login_col1, login_col2, login_col3 = st.columns([1, 1.4, 1])

    with login_col2:
        st.markdown("""
            <div class="glass-card" style="padding: 28px; position: relative; z-index: 20;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="color: #ffffff; font-size: 18px; margin: 0;">🔐 Role-Based Access</h3>
                    <span style="font-size: 11px; background: rgba(0, 240, 255, 0.15); color: #00f0ff; border: 1px solid rgba(0, 240, 255, 0.4); padding: 2px 8px; border-radius: 4px; font-weight: 700;">SECURE SSL</span>
                </div>
                <p style="color: #94a3b8; font-size: 13px; margin-bottom: 18px;">
                    Select an executive or staff role to load the designated operational workspace.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("#### ⚡ 1-Click Demo Login")
        q1, q2 = st.columns(2)
        with q1:
            if st.button("👑 Shop Owner\n(Full Executive Access)", use_container_width=True):
                st.session_state.auth_user = USERS_DB["owner"]
                st.rerun()
        with q2:
            if st.button("👷 Floor Staff\n(Stock Details & Queue)", use_container_width=True):
                st.session_state.auth_user = USERS_DB["rahul"]
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🔑 Or Enter Credentials")
        with st.form("login_form"):
            uname = st.text_input("Username", placeholder="e.g. owner or rahul or priya").strip().lower()
            pwd = st.text_input("Password", type="password", placeholder="Enter password (default: 123)")
            submitted = st.form_submit_button("Sign In to Workspace", use_container_width=True)

            if submitted:
                if uname in USERS_DB and USERS_DB[uname]["password"] == pwd:
                    st.session_state.auth_user = USERS_DB[uname]
                    st.success(f"Welcome back, {USERS_DB[uname]['display_name']}!")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("Invalid credentials. (Hint: Use 'owner' or 'rahul' with password '123')")

    st.stop()


# ==============================================================================
# 7. LOGGED-IN SESSION SIDEBAR & ROLE ROUTING
# ==============================================================================
curr_user = st.session_state.auth_user
is_owner = (curr_user["role"] == "Shop Owner")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart--v1.png", width=55)
    st.markdown("### **Everseen Portal**")
    
    badge_html = f'<span class="{curr_user["badge_class"]}">{curr_user["icon"]} {curr_user["role"].upper()}</span>'
    st.markdown(f"""
        <div style="background: rgba(10, 17, 36, 0.88); border: 1px solid rgba(0, 240, 255, 0.3); padding: 12px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="color: #ffffff; font-size: 13px;">{curr_user['display_name']}</strong>
            </div>
            <div style="margin-top: 6px;">{badge_html}</div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{curr_user['email']}</div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.auth_user = None
        st.rerun()

    st.markdown("---")

    if is_owner:
        st.markdown("👑 **Owner Executive Controls**")
        app_mode = st.radio(
            "Dashboard View",
            [
                "📹 Live CCTV Operations",
                "🗺️ Store Floor Plan & Aisle Heatmap",
                "🧠 Temporal Persistence Filter (AI)",
                "📐 Planogram Digital Twin",
                "💰 Financial ROI Calculator",
                "📊 Compliance Analytics",
                "📦 Full Stock & Inventory Master",
                "🚨 Incident Audit Trail",
                "🏗️ Architecture Blueprint"
            ]
        )
    else:
        st.markdown("👷 **Floor Operations Menu**")
        app_mode = st.radio(
            "Staff View",
            [
                "📋 My Restock Task Queue",
                "📦 Live Stock & Inventory Inspector",
                "📍 Shelf Planogram & Facing Guide",
                "🔍 Quick SKU Barcode Scanner"
            ]
        )

    st.markdown("---")
    st.markdown("### 🎥 Camera Stream Switcher")
    selected_camera = st.selectbox(
        "Active Camera Feed",
        [
            "CAM-01: Aisle 3 (Beverages & Soda)",
            "CAM-02: Aisle 1 (Snacks & Confectionery)",
            "CAM-03: Aisle 2 (Dairy & Refrigerated)",
            "CAM-04: Aisle 4 (Personal Care & Hygiene)"
        ]
    )

    if is_owner:
        st.markdown("---")
        st.markdown("### ⚡ Live Demo Scenarios")
        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            if st.button("🚨 Void Spike", use_container_width=True):
                aisle_name = selected_camera.split(":")[1].split("(")[0].strip() + ", Shelf B"
                api_simulate_alert(aisle_name, "Pepsi Zero 500ml", f"SKU #{np.random.randint(2000, 5000)}")
                st.toast("🚨 Stockout alert dispatched to staff handheld!", icon="⚠️")
                time.sleep(0.3)
                st.rerun()
        with col_sc2:
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state.fallback_active = []
                st.toast("Active alerts cleared for demo.", icon="🧹")
                time.sleep(0.3)
                st.rerun()


# ==============================================================================
# 8. EMPLOYEE DASHBOARD VIEWS (Stock Details & Restock Focus)
# ==============================================================================
if not is_owner:
    if app_mode == "📋 My Restock Task Queue":
        st.markdown(f"""
            <div class="noc-header">
                <div>
                    <p class="noc-title">📋 Floor Restock Assignment Queue</p>
                    <p class="noc-subtitle">Active void alerts routed to your shift • Pick up items and restock designated shelf facings</p>
                </div>
                <div>
                    <span class="role-badge-employee">LOGGED IN: {curr_user['display_name']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col_tasks, col_feed = st.columns([1.8, 1.2])

        with col_tasks:
            st.subheader("⚠️ Priority Stockout Voids Requiring Restock")
            active_alerts, resolved_alerts = api_get_alerts()

            if not active_alerts:
                st.success("🎉 Great job! No pending stockout alerts assigned to your aisles.")
            
            for alert in active_alerts:
                st.markdown(f"""
                    <div class="alert-item-critical">
                        <div style="display: flex; justify-content: space-between;">
                            <strong style="color: #ffffff; font-size: 15px;">📍 {alert.get('aisle')}</strong>
                            <span style="background-color: #ef4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 800;">PRIORITY P1</span>
                        </div>
                        <div style="font-size: 14px; color: #f8fafc; margin-top: 6px;">
                            <strong>Product:</strong> {alert.get('product_name')} ({alert.get('sku')})
                        </div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 3px;">
                            Void Duration: <span style="color: #facc15; font-family: monospace;">{alert.get('duration_sec', 0)}s</span> • Assigned: {curr_user['display_name']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                c_res1, c_res2 = st.columns([2, 1])
                with c_res1:
                    st.info(f"👉 Recommended Action: Fetch 12 units of {alert.get('product_name')} from Backroom Rack B-4.")
                with c_res2:
                    if st.button("✅ Mark Restocked", key=f"emp_res_{alert.get('alert_id')}", use_container_width=True):
                        api_resolve_alert(alert.get("alert_id"), curr_user["display_name"])
                        st.toast("✅ Stockout marked as resolved! System will re-scan shelf.", icon="🎉")
                        time.sleep(0.3)
                        st.rerun()

        with col_feed:
            st.subheader("📹 Live CCTV Verification")
            current_frame = render_enhanced_shelf_frame(selected_camera, 0.6, True, True)
            st.image(current_frame, channels="BGR", caption=f"Real-time visual check for {selected_camera}", use_container_width=True)

    elif app_mode == "📦 Live Stock & Inventory Inspector":
        st.markdown(f"""
            <div class="noc-header">
                <div>
                    <p class="noc-title">📦 Live Retail Inventory & SKU Master Details</p>
                    <p class="noc-subtitle">Real-time stock levels, min-order thresholds, unit prices, and physical aisle locations</p>
                </div>
                <div>
                    <span class="role-badge-employee">STAFF PORTAL</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        f1, f2, f3 = st.columns(3)
        with f1:
            sel_cat = st.selectbox("Filter Category", ["All Categories", "Beverages", "Energy Drinks", "Snacks", "Dairy", "Personal Care"])
        with f2:
            sel_status = st.selectbox("Stock Status", ["All Items", "In Stock", "OUT OF STOCK", "LOW STOCK"])
        with f3:
            search_sku = st.text_input("Search SKU or Name", placeholder="e.g. Coca-Cola or SKU #202")

        df_stock = pd.DataFrame(INVENTORY_STOCK_DB)
        if sel_cat != "All Categories":
            df_stock = df_stock[df_stock["category"] == sel_cat]
        if sel_status == "OUT OF STOCK":
            df_stock = df_stock[df_stock["in_stock"] == 0]
        elif sel_status == "LOW STOCK":
            df_stock = df_stock[(df_stock["in_stock"] > 0) & (df_stock["in_stock"] <= df_stock["min_threshold"])]
        if search_sku:
            df_stock = df_stock[df_stock["name"].str.contains(search_sku, case=False) | df_stock["sku"].str.contains(search_sku, case=False)]

        st.dataframe(df_stock, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📋 Quick Stock Refill Request Form")
        with st.form("refill_form"):
            refill_sku = st.selectbox("Select SKU needing backroom pull", [f"{x['sku']} - {x['name']}" for x in INVENTORY_STOCK_DB])
            refill_qty = st.number_input("Cases / Units to request from warehouse", min_value=1, max_value=100, value=12)
            if st.form_submit_button("Submit Warehouse Refill Ticket"):
                st.toast(f"Ticket filed for {refill_qty} units of {refill_sku}!", icon="📦")

    elif app_mode == "📍 Shelf Planogram & Facing Guide":
        st.markdown("""
            <div class="noc-header">
                <div>
                    <p class="noc-title">📍 Shelf Facing & Planogram Restock Guide</p>
                    <p class="noc-subtitle">Exact visual layout of where each product must be placed according to store planogram</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.info("💡 Follow this slot sequence to ensure shelves match official brand contracts and planograms.")

        st.markdown("#### **Shelf A (Top Rack - Colas & Sodas)**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Slot 1", "Coca-Cola 500ml", "4 Facings")
        c2.metric("Slot 2", "Diet Coke 500ml", "4 Facings")
        c3.metric("Slot 3", "Sprite 500ml", "4 Facings")
        c4.metric("Slot 4", "Fanta Orange", "4 Facings")
        c5.metric("Slot 5", "Thums Up 500ml", "4 Facings")
        c6.metric("Slot 6", "Mountain Dew", "4 Facings")

        st.markdown("#### **Shelf B (Middle Eye-Level Rack - High Velocity)**")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Slot 1", "Red Bull 250ml", "4 Facings")
        b2.metric("Slot 2", "Pepsi Zero 500ml", "⚠️ 0 / 4 (EMPTY)")
        b3.metric("Slot 3", "Monster Energy", "⚠️ Misplaced")
        b4.metric("Slot 4", "Gatorade 500ml", "4 Facings")
        b5.metric("Slot 5", "Tropicana Orange", "4 Facings")
        b6.metric("Slot 6", "Minute Maid", "4 Facings")

        st.markdown("#### **Shelf C (Bottom Rack - Heavy & Bulk)**")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Slot 1", "Bisleri 1L", "6 Facings")
        k2.metric("Slot 2", "Aquafina 1L", "6 Facings")
        k3.metric("Slot 3", "Kinley 1L", "6 Facings")
        k4.metric("Slot 4", "Himalayan 1L", "4 Facings")
        k5.metric("Slot 5", "Real Juice 1L", "4 Facings")
        k6.metric("Slot 6", "Paper Boat", "4 Facings")

    elif app_mode == "🔍 Quick SKU Barcode Scanner":
        st.markdown("""
            <div class="noc-header">
                <div>
                    <p class="noc-title">🔍 Handheld SKU Barcode Lookup & Scanner</p>
                    <p class="noc-subtitle">Scan or enter barcode number to look up shelf position and warehouse stock</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col_s1, col_s2 = st.columns([1, 1.5])
        with col_s1:
            entered_code = st.text_input("Enter 13-digit EAN Barcode", value="8901030382026")
            if st.button("Simulate Barcode Gun Scan"):
                st.toast("Barcode scanned successfully!", icon="🔫")

        with col_s2:
            matched_item = next((item for item in INVENTORY_STOCK_DB if item["barcode"] == entered_code), None)
            if matched_item:
                status_color = "#ef4444" if matched_item["status"] == "OUT OF STOCK" else "#00f0ff"
                st.markdown(f"""
                    <div class="glass-card" style="border-left: 6px solid {status_color};">
                        <div style="font-size: 12px; color: #94a3b8;">BARCODE MATCH FOUND:</div>
                        <h3 style="color: #ffffff; margin: 4px 0;">{matched_item['name']}</h3>
                        <p style="color: #00f0ff; font-size: 14px; font-weight: 600;">{matched_item['sku']} • Category: {matched_item['category']}</p>
                        <hr style="border-color: #334155; margin: 10px 0;">
                        <p><strong>Physical Location:</strong> {matched_item['aisle']} → {matched_item['shelf']}</p>
                        <p><strong>Retail Price:</strong> {matched_item['unit_price']}</p>
                        <p><strong>Shelf Stock Count:</strong> <span style="color: {status_color}; font-weight: 800;">{matched_item['in_stock']} Units</span> (Min: {matched_item['min_threshold']})</p>
                        <p><strong>Status:</strong> <span style="color: {status_color};">{matched_item['status']}</span></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No item found matching this barcode. Please verify barcode number.")

    st.stop()


# ==============================================================================
# 9. SHOP OWNER DASHBOARD VIEWS (Full Executive Access)
# ==============================================================================
if app_mode == "📹 Live CCTV Operations":
    if backend_online:
        backend_badge = f'<span style="color: #00f0ff; font-weight: 700;">● FASTAPI CONNECTED (v{backend_meta.get("version", "2.0.0")})</span>'
    else:
        backend_badge = '<span style="color: #f59e0b; font-weight: 700;">● LOCAL STANDALONE MODE</span>'

    st.markdown(f"""
        <div class="noc-header">
            <div>
                <p class="noc-title">👁️ Executive CCTV Operations & Real-Time Void Analysis</p>
                <p class="noc-subtitle">Automated shelf auditing and restock orchestration powered by Everseen pattern</p>
            </div>
            <div style="text-align: right;">
                <div class="rec-indicator"><span class="pulse-dot"></span> LIVE INFERENCE</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-top: 6px;">{backend_badge}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Live Aisle Compliance</div>
                <div style="font-size: 26px; font-weight: 800; color: #00f0ff; font-family: monospace;">94.2%</div>
                <div style="font-size: 12px; color: #4ade80;">↑ +1.8% vs baseline</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Active Stockout Voids</div>
                <div style="font-size: 26px; font-weight: 800; color: #f87171; font-family: monospace;">2 Slots</div>
                <div style="font-size: 12px; color: #f87171;">Requires restocker action</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Avg Restock SLA</div>
                <div style="font-size: 26px; font-weight: 800; color: #38bdf8; font-family: monospace;">3.4m</div>
                <div style="font-size: 12px; color: #4ade80;">↓ 45s faster response</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Revenue Saved Today</div>
                <div style="font-size: 26px; font-weight: 800; color: #22c55e; font-family: monospace;">₹24,850</div>
                <div style="font-size: 12px; color: #4ade80;">↑ +₹6,200 via prompt restock</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    v_mode = st.radio("CCTV Layout Mode", ["Single Camera Focus", "Quad Multi-Camera Matrix (4 Aisles Live)"], horizontal=True)

    if v_mode == "Quad Multi-Camera Matrix (4 Aisles Live)":
        st.markdown("### 🖥️ Storewide 4-Camera Multi-View Operations Grid")
        q_row1_c1, q_row1_c2 = st.columns(2)
        with q_row1_c1:
            st.caption("🟢 CAM-01: Aisle 3 (Beverages & Soda) - 1080p | 30 FPS")
            f1 = render_enhanced_shelf_frame("CAM-01: AISLE 3 (BEV)", 0.6, True, True)
            st.image(f1, channels="BGR", use_container_width=True)
        with q_row1_c2:
            st.caption("🟢 CAM-02: Aisle 1 (Snacks & Chips) - 1080p | 30 FPS")
            f2 = render_enhanced_shelf_frame("CAM-02: AISLE 1 (SNACKS)", 0.6, True, False)
            st.image(f2, channels="BGR", use_container_width=True)

        q_row2_c1, q_row2_c2 = st.columns(2)
        with q_row2_c1:
            st.caption("🟢 CAM-03: Aisle 2 (Dairy & Refrigerated) - 1080p | 25 FPS")
            f3 = render_enhanced_shelf_frame("CAM-03: AISLE 2 (DAIRY)", 0.6, True, False)
            st.image(f3, channels="BGR", use_container_width=True)
        with q_row2_c2:
            st.caption("🟢 CAM-04: Aisle 4 (Personal Care) - 1080p | 30 FPS")
            f4 = render_enhanced_shelf_frame("CAM-04: AISLE 4 (CARE)", 0.6, True, True)
            st.image(f4, channels="BGR", use_container_width=True)

    else:
        col_vid, col_alert = st.columns([1.8, 1.2])

        with col_vid:
            st.markdown(f"""
                <div class="stream-status-bar">
                    <span>RTSP FEED: {selected_camera}</span>
                    <span>CODEC: H.264 | 1080p | 30 FPS</span>
                    <span style="color: #00f0ff;">● SECURE RTSP</span>
                </div>
            """, unsafe_allow_html=True)

            current_frame = render_enhanced_shelf_frame(selected_camera, 0.6, True, True)
            st.image(current_frame, channels="BGR", use_container_width=True)

            ctrl1, ctrl2 = st.columns(2)
            with ctrl1:
                auto_stream = st.toggle("🔴 Stream Simulation Loop", value=False)
                if auto_stream:
                    time.sleep(1)
                    st.rerun()
            with ctrl2:
                if st.button("📸 Capture Shelf Frame", use_container_width=True):
                    st.toast("CCTV frame archived to /data/raw_frames/CAM03_audit.jpg", icon="💾")

        with col_alert:
            st.markdown("### ⚠️ Real-Time Alert Operations Center")
            tab_active, tab_resolved = st.tabs(["🔴 Active Stockouts", "✅ Resolved Feeds"])
            active_alerts, resolved_alerts = api_get_alerts()

            with tab_active:
                if not active_alerts:
                    st.success("🎉 All shelves are fully compliant. No active stockout voids detected!")
                
                for alert in active_alerts:
                    severity = alert.get("severity", "critical")
                    card_class = "alert-item-critical" if severity == "critical" else "alert-item-warning"
                    badge = "🔴 CRITICAL STOCKOUT" if severity == "critical" else "🟡 MISPLACED SKU"

                    st.markdown(f"""
                        <div class="{card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="color: #ffffff; font-size: 14px;">{alert.get('aisle')}</strong>
                                <span style="font-size: 11px; font-weight: 800; color: #f87171;">{badge}</span>
                            </div>
                            <div style="margin-top: 6px; font-size: 13px; color: #e2e8f0;">
                                <strong>{alert.get('product_name')}</strong> ({alert.get('sku')})
                            </div>
                            <div style="font-size: 12px; color: #94a3b8; margin-top: 3px;">
                                Void Duration: <span style="color: #facc15; font-family: monospace;">{alert.get('duration_sec', 0)}s</span> • SLA Target: 5 mins
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    act_col1, act_col2 = st.columns([1.5, 1])
                    with act_col1:
                        staff = st.selectbox(
                            "Assign Restocker",
                            ["Rahul Verma (Staff #1)", "Priya Sharma (Staff #2)", "Amit Patel (Staff #3)", "Ramesh K (Staff #4)"],
                            key=f"owner_staff_{alert.get('alert_id')}"
                        )
                    with act_col2:
                        st.write("")
                        if st.button("Mark Restocked", key=f"res_owner_{alert.get('alert_id')}", use_container_width=True):
                            api_resolve_alert(alert.get("alert_id"), staff)
                            st.toast(f"✅ Alert {alert.get('alert_id')} marked as resolved by {staff}!", icon="🎉")
                            time.sleep(0.3)
                            st.rerun()

            with tab_resolved:
                for alert in resolved_alerts:
                    st.markdown(f"""
                        <div class="alert-item-success">
                            <strong style="color: #ffffff;">{alert.get('aisle')}</strong> — <span style="color: #4ade80;">Resolved</span><br>
                            <span style="font-size: 13px; color: #e2e8f0;">{alert.get('product_name')} ({alert.get('sku')})</span><br>
                            <small style="color: #86efac;">✅ Restocked by {alert.get('assigned_to', 'Staff')}</small>
                        </div>
                    """, unsafe_allow_html=True)

elif app_mode == "🗺️ Store Floor Plan & Aisle Heatmap":
    st.markdown("""
        <div class="noc-header">
            <div>
                <p class="noc-title">🗺️ Store Digital Twin — 2D Floor Plan & Anomaly Heatmap</p>
                <p class="noc-subtitle">Spatial tracking showing active stockout locations across supermarket corridors</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    map_c1, map_c2 = st.columns([2, 1])

    with map_c1:
        st.markdown("### 🏬 Supermarket Floor Layout (Top-Down View)")
        st.caption("Click any aisle corridor button to review its spatial compliance status.")

        st.markdown("""
            <div class="store-map-container">
                <div style="display: flex; justify-content: space-between; border-bottom: 2px dashed #334155; padding-bottom: 8px; margin-bottom: 14px;">
                    <span style="color: #94a3b8; font-size: 12px; font-weight: 700;">🚪 STORE ENTRANCE / TURNSTILES</span>
                    <span style="color: #94a3b8; font-size: 12px; font-weight: 700;">🧾 CHECKOUT REGISTERS (POS 1-6)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        m_row1_c1, m_row1_c2 = st.columns(2)
        with m_row1_c1:
            st.markdown("""
                <div style="background: rgba(225, 29, 72, 0.15); border: 2px solid #ef4444; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #f87171; font-size: 14px;">AISLE 3: BEVERAGES & SODAS</strong>
                        <span style="background: #ef4444; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 800;">🔴 2 VOIDS</span>
                    </div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px;">
                        CAM-01 Active • Missing: Coca-Cola 500ml & Pepsi Zero
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("🔎 Jump to Aisle 3 CCTV Feed"):
                st.toast("Switched target camera to CAM-01 (Aisle 3)", icon="📹")

        with m_row1_c2:
            st.markdown("""
                <div style="background: rgba(245, 158, 11, 0.15); border: 2px solid #f59e0b; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #f59e0b; font-size: 14px;">AISLE 1: SNACKS & CONFECTIONERY</strong>
                        <span style="background: #f59e0b; color: black; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 800;">🟡 1 MISPLACED</span>
                    </div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px;">
                        CAM-02 Active • Anomaly: Lays Onion misplaced slot
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("🔎 Jump to Aisle 1 CCTV Feed"):
                st.toast("Switched target camera to CAM-02 (Aisle 1)", icon="📹")

        m_row2_c1, m_row2_c2 = st.columns(2)
        with m_row2_c1:
            st.markdown("""
                <div style="background: rgba(34, 197, 94, 0.15); border: 2px solid #22c55e; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #4ade80; font-size: 14px;">AISLE 2: DAIRY & CHILLED</strong>
                        <span style="background: #22c55e; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 800;">🟢 100% OK</span>
                    </div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px;">
                        CAM-03 Active • Full compliance verified
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with m_row2_c2:
            st.markdown("""
                <div style="background: rgba(34, 197, 94, 0.15); border: 2px solid #22c55e; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #4ade80; font-size: 14px;">AISLE 4: PERSONAL CARE</strong>
                        <span style="background: #22c55e; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 800;">🟢 COMPLIANT</span>
                    </div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px;">
                        CAM-04 Active • Restocked 12m ago
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align: center; border-top: 2px dashed #334155; padding-top: 10px; margin-top: 10px; font-size: 12px; color: #94a3b8;">
                📦 BACKROOM STORAGE & WAREHOUSE STAGING BAY
            </div>
        """, unsafe_allow_html=True)

    with map_c2:
        st.markdown("### 📊 Corridor Congestion & Void Density")
        st.metric("Highest Risk Zone", "Aisle 3 (Beverages)", "38% of store vacancies")
        st.metric("Avg Customer Dwell Time", "4.2 mins", "High engagement")
        st.metric("Restock Distance", "18 meters", "From warehouse bay")

elif app_mode == "🧠 Temporal Persistence Filter (AI)":
    st.markdown("""
        <div class="noc-header">
            <div>
                <p class="noc-title">🧠 Temporal Persistence Tracker & Occlusion Engine</p>
                <p class="noc-subtitle">Explainable AI: How computer vision prevents false alarms when shoppers pick or browse items</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    In high-traffic retail environments, customers constantly reach for items, pick products up to inspect nutrition labels, or briefly block camera view lines. **A naive YOLO detector would falsely trigger stockout alarms thousands of times per day.**
    
    The **Everseen Temporal Persistence Engine** tracks bounding box vacancies across continuous temporal windows before elevating an anomaly.
    """)

    st.markdown("---")
    st.markdown("### 🕹️ Interactive Temporal Window Simulator")

    t_sec = st.slider("Simulate Duration of Empty Shelf State (Seconds)", 0, 60, st.session_state.temporal_test_sec)
    st.session_state.temporal_test_sec = t_sec

    col_sim1, col_sim2, col_sim3 = st.columns(3)

    is_phase1 = (t_sec <= 10)
    is_phase2 = (t_sec > 10 and t_sec < 30)
    is_phase3 = (t_sec >= 30)

    with col_sim1:
        cls1 = "timeline-step step-active" if is_phase1 else "timeline-step"
        st.markdown(f"""
            <div class="{cls1}">
                <div style="font-size: 12px; color: #38bdf8; font-weight: 700;">STAGE 1: 0s – 10s</div>
                <h4 style="color: #ffffff; margin: 4px 0;">Transient Interaction</h4>
                <p style="font-size: 12px; color: #94a3b8;">Customer hand occlusion or brief product inspection.</p>
                <span style="color: #4ade80; font-weight: 700; font-size: 12px;">STATUS: SUPPRESSED (NO ALARM)</span>
            </div>
        """, unsafe_allow_html=True)

    with col_sim2:
        cls2 = "timeline-step step-active" if is_phase2 else "timeline-step"
        st.markdown(f"""
            <div class="{cls2}">
                <div style="font-size: 12px; color: #f59e0b; font-weight: 700;">STAGE 2: 11s – 29s</div>
                <h4 style="color: #ffffff; margin: 4px 0;">Evaluating Persistence</h4>
                <p style="font-size: 12px; color: #94a3b8;">Void persisted across 300+ video frames. Verifying customer departure.</p>
                <span style="color: #f59e0b; font-weight: 700; font-size: 12px;">STATUS: ARMED / COUNTDOWN</span>
            </div>
        """, unsafe_allow_html=True)

    with col_sim3:
        cls3 = "timeline-step step-active" if is_phase3 else "timeline-step"
        st.markdown(f"""
            <div class="{cls3}">
                <div style="font-size: 12px; color: #ef4444; font-weight: 700;">STAGE 3: 30s+</div>
                <h4 style="color: #ffffff; margin: 4px 0;">Confirmed Out-Of-Stock</h4>
                <p style="font-size: 12px; color: #94a3b8;">True retail shelf vacancy confirmed with 99.4% precision.</p>
                <span style="color: #ef4444; font-weight: 700; font-size: 12px;">STATUS: DISPATCH P1 RESTOCK ALERT</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    if is_phase3:
        st.error(f"🚨 Temporal Threshold Exceeded ({t_sec}s >= 30s)! Real-time stockout alert dispatched to staff handheld.")
    elif is_phase2:
        st.warning(f"⏳ Tracking vacancy ({t_sec}s / 30s). False alarm suppression active.")
    else:
        st.success(f"🟢 Normal customer browsing interaction detected ({t_sec}s). Suppressed by ByteTrack.")

elif app_mode == "📐 Planogram Digital Twin":
    st.markdown("""
        <div class="noc-header">
            <div>
                <p class="noc-title">📐 Planogram Digital Twin & Heatmap Verification</p>
                <p class="noc-subtitle">Executive matrix verifying planogram compliance and physical facings</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Aisle Planogram ID", "PG-AISLE3-BEV-v4")
    p2.metric("Total Facing Capacity", "18 Facings")
    p3.metric("Planogram Compliance", "88.9%", "-5.5% vs Target")
    p4.metric("Misplaced Item Count", "1 Item", "Needs realignment")

    st.markdown("### 🛒 Shelf Facings Matrix (Expected vs Real-Time CV)")
    st.info("The CV engine applies 4-point perspective warping to map oblique retail camera views into this normalized planogram space.")

elif app_mode == "💰 Financial ROI Calculator":
    st.markdown("""
        <div class="noc-header">
            <div>
                <p class="noc-title">💰 Store Revenue Optimization & Financial ROI Simulator</p>
                <p class="noc-subtitle">Executive financial modeling for store revenue protection</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c_in, c_out = st.columns([1.2, 1.8])
    with c_in:
        daily_footfall = st.slider("Daily Store Shopper Footfall", 500, 15000, 3200, 100)
        avg_basket_val = st.slider("Average Item Value (₹)", 20, 500, 75, 5)
        stockout_rate = st.slider("Industry Baseline Out-of-Stock Rate (%)", 3.0, 15.0, 8.2, 0.2)
        recovery_efficiency = st.slider("Everseen Automated Recovery Rate (%)", 40, 95, 78, 2)

    with c_out:
        daily_potential_loss = daily_footfall * (stockout_rate / 100.0) * avg_basket_val * 0.45
        daily_recovered = daily_potential_loss * (recovery_efficiency / 100.0)
        annual_recovered = daily_recovered * 365
        annual_recovered_lakhs = annual_recovered / 100000.0

        r1, r2, r3 = st.columns(3)
        r1.metric("Daily Revenue Saved", f"₹{int(daily_recovered):,}")
        r2.metric("Monthly Net Impact", f"₹{int(daily_recovered * 30):,}")
        r3.metric("Annual ROI Projection", f"₹{annual_recovered_lakhs:.2f} Lakhs")

elif app_mode == "📊 Compliance Analytics":
    st.markdown("""
        <div class="noc-header">
            <div>
                <p class="noc-title">📊 Operational Compliance & Inventory Analytics</p>
                <p class="noc-subtitle">Real-time breakdown of stockout trends, peak shopping hours, and staff velocity</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### 📈 Hourly Out-of-Stock Spikes (Today)")
        hours = [f"{h}:00" for h in range(8, 22)]
        chart_data = pd.DataFrame({
            "Hour": hours,
            "Beverages & Soda": [3, 4, 6, 9, 14, 16, 12, 8, 10, 15, 17, 9, 5, 2],
            "Snacks & Confectionery": [1, 2, 4, 7, 10, 12, 9, 6, 8, 11, 13, 7, 4, 1],
            "Dairy & Chilled": [2, 3, 3, 2, 3, 4, 2, 2, 3, 4, 5, 3, 2, 1]
        }).set_index("Hour")
        st.area_chart(chart_data)

    with g2:
        st.markdown("### 🏆 Restock Staff Performance & SLA Leaderboard")
        staff_df = pd.DataFrame({
            "Staff Member": ["Rahul Verma (Staff #1)", "Priya Sharma (Staff #2)", "Amit Patel (Staff #3)", "Ramesh K (Staff #4)"],
            "Assigned Tasks": [18, 15, 12, 14],
            "Avg Resolution Time": ["2.8 mins", "3.1 mins", "3.7 mins", "4.0 mins"],
            "SLA Compliance": ["98.4%", "95.2%", "92.0%", "90.5%"],
            "Rating": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐"]
        })
        st.dataframe(staff_df, use_container_width=True, hide_index=True)

elif app_mode == "📦 Full Stock & Inventory Master":
    st.markdown("""
        <div class="noc-header">
            <div>
                <p class="noc-title">📦 Master Retail Inventory Database</p>
                <p class="noc-subtitle">Owner view: full audit of all SKUs, in-stock levels, supplier thresholds, and barcodes</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    df_full = pd.DataFrame(INVENTORY_STOCK_DB)
    st.dataframe(df_full, use_container_width=True, hide_index=True)

elif app_mode == "🚨 Incident Audit Trail":
    st.markdown("""
        <div class="noc-header">
            <div>
                <p class="noc-title">🚨 Store Incident Audit Trail & Compliance Export</p>
                <p class="noc-subtitle">Complete historical incident logs with one-click export for store operations</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    audit_data = [
        {"Timestamp": "2026-09-28 09:34:10", "Incident ID": "INC-8891", "Aisle": "Aisle 3", "SKU": "SKU #4589", "Product": "Coca-Cola 500ml", "Type": "Stockout", "Severity": "Critical", "Duration": "38s", "Status": "Pending"},
        {"Timestamp": "2026-09-28 09:28:45", "Incident ID": "INC-8890", "Aisle": "Aisle 1", "SKU": "SKU #1102", "Product": "Lays Cream & Onion", "Type": "Misplaced", "Severity": "Warning", "Duration": "1m 15s", "Status": "Pending"},
        {"Timestamp": "2026-09-28 09:15:20", "Incident ID": "INC-8889", "Aisle": "Aisle 2", "SKU": "SKU #8820", "Product": "Amul Butter 500g", "Type": "Stockout", "Severity": "Resolved", "Duration": "2m 10s", "Status": "Resolved"},
    ]
    st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)

else:
    st.markdown("""
        <div class="noc-header">
            <div>
                <p class="noc-title">🏗️ End-to-End Edge Architecture (Everseen Pattern)</p>
                <p class="noc-subtitle">System design showing camera pipeline, homography, and real-time alert broker</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ```
    ┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
    │  Retail CCTV Camera      │      │  Homography Calibration │      │  YOLOv8 Detection        │
    │  • 1080p RTSP Stream     │ ───► │  • 4-Point Transformation│ ───► │  • SKU Bounding Boxes    │
    │  • Oblique Shelf Angle   │      │  • Normalized Planogram  │      │  • Void / Empty Slot Reg │
    └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
                                                                                      │
                                                                                      ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
    │  Role-Based Dashboards   │      │  FastAPI Real-Time Broker│      │  Temporal Persistence    │
    │  • Shop Owner (Full NOC) │ ◄─── │  • REST & WebSockets     │ ◄─── │  • 30s Delay Filter      │
    │  • Floor Staff (Stock Ops│      │  • In-Memory Event State │      │  • Occlusion Filter      │
    └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
    ```
    """)
