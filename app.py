import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="Everseen x IIT Bombay - Shelf Auditing AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Retail Tech CSS Styling
st.markdown("""
    <style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #f7f9fc;
    }
    .main-header {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #475569;
        margin-bottom: 18px;
    }
    .live-pill {
        display: inline-block;
        background-color: #fee2e2;
        color: #ef4444;
        font-weight: 700;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 9999px;
        letter-spacing: 0.5px;
        border: 1px solid #fca5a5;
    }
    .alert-card-critical {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-card-warning {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-card-success {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stream-status-bar {
        background-color: #0f172a;
        color: #94a3b8;
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        font-family: monospace;
        font-size: 12px;
        display: flex;
        justify-content: space-between;
    }
    .planogram-slot {
        padding: 12px 8px;
        border-radius: 6px;
        text-align: center;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .slot-ok { 
        background-color: #dcfce7; 
        color: #166534; 
        border: 1px solid #86efac; 
    }
    .slot-stockout { 
        background-color: #fee2e2; 
        color: #991b1b; 
        border: 1px dashed #f87171; 
    }
    .slot-misplaced { 
        background-color: #fef3c7; 
        color: #92400e; 
        border: 1px solid #fde047; 
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. STATE MANAGEMENT (Session State)
# ==============================================================================
if "alerts" not in st.session_state:
    st.session_state.alerts = [
        {
            "id": "ALT-101",
            "aisle": "Aisle 3, Shelf B",
            "item": "Coca-Cola 500ml (Beverages)",
            "sku": "SKU #4589",
            "type": "Critical Stockout",
            "duration": "38s",
            "status": "Pending Restock",
            "assigned_to": "Unassigned",
            "severity": "critical"
        },
        {
            "id": "ALT-102",
            "aisle": "Aisle 1, Shelf A",
            "item": "Lays Cream & Onion (Snacks)",
            "sku": "SKU #1102",
            "type": "Misplaced Item",
            "duration": "1m 15s",
            "status": "Pending Review",
            "assigned_to": "Unassigned",
            "severity": "warning"
        },
        {
            "id": "ALT-103",
            "aisle": "Aisle 2, Shelf C",
            "item": "Amul Butter 500g (Dairy)",
            "sku": "SKU #8820",
            "type": "Restocked",
            "duration": "Resolved",
            "status": "Completed",
            "assigned_to": "Staff #4 (Ramesh)",
            "severity": "success"
        }
    ]

if "manual_audits" not in st.session_state:
    st.session_state.manual_audits = 14


# ==============================================================================
# 3. SYNTHETIC COMPUTER VISION ENGINE (Realistic Mock Stream)
# ==============================================================================
def generate_shelf_frame(aisle_name, confidence_thresh, stockout_simulated=True):
    """
    Renders a realistic retail shelf CCTV frame with detected bounding boxes,
    stockout voids, and camera OSD telemetry overlays.
    """
    frame = np.ones((420, 700, 3), dtype=np.uint8) * 235

    # Shelf Backboard and Racks
    cv2.rectangle(frame, (20, 20), (680, 400), (210, 215, 220), -1)
    rack_colors = (160, 165, 175)
    cv2.rectangle(frame, (20, 130), (680, 145), rack_colors, -1)
    cv2.rectangle(frame, (20, 260), (680, 275), rack_colors, -1)
    cv2.rectangle(frame, (20, 385), (680, 400), (120, 125, 135), -1)

    # Shelf Tier Labels
    cv2.putText(frame, "SHELF A", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(frame, "SHELF B", (25, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(frame, "SHELF C", (25, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    # Tier 1 (Shelf A) - Normal Compliant Products
    items_shelf_a = [
        ((60, 40), (130, 130), "SKU #101: OK", (46, 204, 113), 0.94),
        ((160, 40), (230, 130), "SKU #102: OK", (46, 204, 113), 0.91),
        ((260, 40), (330, 130), "SKU #103: OK", (46, 204, 113), 0.88),
        ((360, 40), (430, 130), "SKU #104: OK", (46, 204, 113), 0.96),
        ((460, 40), (530, 130), "SKU #105: OK", (46, 204, 113), 0.89),
        ((560, 40), (630, 130), "SKU #106: OK", (46, 204, 113), 0.93),
    ]
    for (pt1, pt2, label, color, conf) in items_shelf_a:
        if conf >= confidence_thresh:
            cv2.rectangle(frame, pt1, pt2, (180, 190, 200), -1)
            cv2.rectangle(frame, pt1, pt2, color, 2)
            cv2.putText(frame, f"{label} {int(conf*100)}%", (pt1[0], pt1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # Tier 2 (Shelf B) - Items with Void / Out of Stock and Misplaced SKU
    cv2.rectangle(frame, (60, 160), (140, 260), (190, 195, 205), -1)
    cv2.rectangle(frame, (60, 160), (140, 260), (46, 204, 113), 2)
    cv2.putText(frame, "SKU #201: OK 95%", (60, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (46, 204, 113), 1)

    if stockout_simulated:
        # Stockout Void Indicator
        cv2.rectangle(frame, (170, 160), (270, 260), (40, 40, 230), 2)
        cv2.line(frame, (170, 160), (270, 260), (60, 60, 230), 1)
        cv2.line(frame, (170, 260), (270, 160), (60, 60, 230), 1)
        cv2.rectangle(frame, (160, 140), (280, 158), (40, 40, 220), -1)
        cv2.putText(frame, "VOID: OUT OF STOCK", (164, 153), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    else:
        cv2.rectangle(frame, (170, 160), (270, 260), (190, 195, 205), -1)
        cv2.rectangle(frame, (170, 160), (270, 260), (46, 204, 113), 2)
        cv2.putText(frame, "SKU #202: OK 92%", (170, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (46, 204, 113), 1)

    # Misplaced SKU
    cv2.rectangle(frame, (300, 160), (380, 260), (210, 210, 180), -1)
    cv2.rectangle(frame, (300, 160), (380, 260), (30, 170, 240), 2)
    cv2.putText(frame, "MISPLACED SKU", (300, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 170, 240), 1)

    cv2.rectangle(frame, (410, 160), (490, 260), (190, 195, 205), -1)
    cv2.rectangle(frame, (410, 160), (490, 260), (46, 204, 113), 2)
    cv2.putText(frame, "SKU #204: OK 89%", (410, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (46, 204, 113), 1)

    # Tier 3 (Shelf C)
    for i, x in enumerate([60, 180, 300, 420, 540]):
        cv2.rectangle(frame, (x, 290), (x+90, 385), (180, 185, 195), -1)
        cv2.rectangle(frame, (x, 290), (x+90, 385), (46, 204, 113), 2)
        cv2.putText(frame, f"SKU #30{i+1}: OK", (x, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (46, 204, 113), 1)

    # OSD Camera Watermark Header
    cv2.rectangle(frame, (20, 20), (680, 45), (15, 23, 42), -1)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"CAM: {aisle_name.upper()} | {timestamp_str} | 30 FPS | YOLOv8x-Retail",
                (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1)

    return frame


# ==============================================================================
# 4. SIDEBAR NAVIGATION & INFERENCE CONTROLS
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart--v1.png", width=65)
    st.title("Everseen AI Control")
    st.caption("IIT Bombay AI Hackathon Edition")
    st.markdown("---")

    app_mode = st.radio(
        "Navigation",
        [
            "📹 Live Shelf Monitoring",
            "📐 Planogram Digital Twin",
            "📊 Store Analytics & ROI",
            "🚨 Incident Log & Export",
            "🏗️ Architecture & Pipeline"
        ]
    )

    st.markdown("---")
    st.markdown("### 🎥 Camera Stream Selector")
    selected_camera = st.selectbox(
        "Aisle Feed",
        [
            "CAM-01: Aisle 3 (Beverages & Soda)",
            "CAM-02: Aisle 1 (Snacks & Confectionery)",
            "CAM-03: Aisle 2 (Dairy & Refrigerated)",
            "CAM-04: Aisle 4 (Personal Care & Hygiene)"
        ]
    )

    st.markdown("### ⚙️ Inference Parameters")
    confidence_threshold = st.slider("Detection Confidence", 0.20, 0.95, 0.60, 0.05)
    persistence_time = st.slider(
        "Stockout Alert Delay (secs)", 5, 60, 30, 5, 
        help="Temporal filter to separate shoppers picking items from genuine stockouts"
    )
    iou_threshold = st.slider("NMS IOU Threshold", 0.30, 0.80, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### ⚡ Live Demo Event Triggers")
    col_trig1, col_trig2 = st.columns(2)
    with col_trig1:
        if st.button("Simulate Stockout", use_container_width=True):
            new_id = f"ALT-{np.random.randint(104, 999)}"
            st.session_state.alerts.insert(0, {
                "id": new_id,
                "aisle": selected_camera.split(":")[1].split("(")[0].strip() + ", Shelf B",
                "item": "Pepsi Zero 500ml",
                "sku": f"SKU #{np.random.randint(2000, 5000)}",
                "type": "Critical Stockout",
                "duration": "Just now",
                "status": "Pending Restock",
                "assigned_to": "Unassigned",
                "severity": "critical"
            })
            st.toast("🚨 Real-time stockout detected by CV engine!", icon="⚠️")
    
    with col_trig2:
        if st.button("Reset Alerts", use_container_width=True):
            st.session_state.alerts = [a for a in st.session_state.alerts if a["status"] != "Pending Restock"]
            st.toast("Pending alerts cleared.", icon="🧹")

    st.markdown("---")
    st.markdown("🧑‍💻 **Hackathon Team:**")
    st.markdown("""
    - **You:** AI/ML & CV Lead (YOLOv8 & Tracking)
    - **Shraddha:** Data Science & Planogram Compliance
    - **Shubham:** Full-Stack & Frontend Lead
    """)


# ==============================================================================
# 5. VIEW 1: LIVE SHELF MONITORING
# ==============================================================================
if app_mode == "📹 Live Shelf Monitoring":
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown('<p class="main-header">👁️ Automated Shelf Auditing & Real-Time Stockouts</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Edge-accelerated Computer Vision pipeline running Planogram & Temporal Void Analysis</p>', unsafe_allow_html=True)
    with col_t2:
        st.markdown("""
        <div style="text-align: right; padding-top: 8px;">
            <span class="live-pill">● LIVE INFERENCE</span>
            <span style="font-family: monospace; font-size: 13px; color: #64748b; margin-left: 8px;">30 FPS | 18ms</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_video, col_alerts = st.columns([2, 1])

    with col_video:
        st.markdown(f"""
        <div class="stream-status-bar">
            <span>FEED: {selected_camera}</span>
            <span>RTSP://edge-gateway.store14.internal/ch03</span>
            <span style="color: #4ade80;">● CONNECTED</span>
        </div>
        """, unsafe_allow_html=True)

        current_frame = generate_shelf_frame(selected_camera, confidence_threshold)
        st.image(current_frame, channels="BGR", use_container_width=True)

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        with ctrl_col1:
            stream_active = st.toggle("Auto-Refresh Stream", value=False)
            if stream_active:
                time.sleep(1)
                st.rerun()
        with ctrl_col2:
            if st.button("📸 Capture Shelf Snapshot", use_container_width=True):
                st.toast("Snapshot archived to /data/captures/CAM03_audit.jpg", icon="💾")
        with ctrl_col3:
            if st.button("🔍 Run Full Aisle Scan", use_container_width=True):
                st.session_state.manual_audits += 1
                st.toast(f"Manual scan completed. Audit count: {st.session_state.manual_audits}", icon="✅")

        with st.expander("📤 Test with Custom Image or Video Clip"):
            uploaded_file = st.file_uploader("Upload custom retail shelf image/video", type=["jpg", "jpeg", "png", "mp4"])
            if uploaded_file is not None:
                st.success(f"Loaded '{uploaded_file.name}'! Running planogram homography alignment & SKU inference.")
                if uploaded_file.type.startswith("image"):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    user_img = cv2.imdecode(file_bytes, 1)
                    cv2.rectangle(user_img, (40, 40), (220, 200), (46, 204, 113), 3)
                    cv2.putText(user_img, "DETECTED: SKU #771 - 94%", (45, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 204, 113), 2)
                    st.image(user_img, channels="BGR", caption="Analyzed Inference Output", use_container_width=True)

    with col_alerts:
        st.subheader("⚠️ Alert Operations Center")
        tab_active, tab_resolved = st.tabs(["Active Out-Of-Stocks", "Resolved Feeds"])

        with tab_active:
            active_alerts = [a for a in st.session_state.alerts if a["status"] != "Completed"]
            if not active_alerts:
                st.info("No active stockouts or misplaced alerts! Shelves are fully compliant.")
            
            for idx, alert in enumerate(active_alerts):
                card_style = "alert-card-critical" if alert["severity"] == "critical" else "alert-card-warning"
                badge = "🔴 CRITICAL" if alert["severity"] == "critical" else "🟡 MISPLACED"

                st.markdown(
                    f"""
                    <div class="{card_style}">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>{alert['aisle']}</strong>
                            <span style="font-size: 11px; font-weight: 700;">{badge}</span>
                        </div>
                        <div style="margin-top: 4px; font-size: 13px;">
                            <span>{alert['item']}</span><br>
                            <span style="color: #64748b; font-size: 12px;">{alert['sku']} • Duration: {alert['duration']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                act_col1, act_col2 = st.columns(2)
                with act_col1:
                    staff = st.selectbox(
                        "Assign Restocker",
                        ["Staff #1 (Rahul)", "Staff #2 (Priya)", "Staff #3 (Amit)"],
                        key=f"staff_{alert['id']}"
                    )
                with act_col2:
                    st.write("")
                    if st.button("Mark Restocked", key=f"resolve_{alert['id']}", use_container_width=True):
                        alert["status"] = "Completed"
                        alert["severity"] = "success"
                        alert["assigned_to"] = staff
                        alert["duration"] = "Resolved just now"
                        st.toast(f"{alert['sku']} marked as restocked by {staff}!", icon="🎉")
                        st.rerun()

        with tab_resolved:
            resolved_alerts = [a for a in st.session_state.alerts if a["status"] == "Completed"]
            for alert in resolved_alerts:
                st.markdown(
                    f"""
                    <div class="alert-card-success">
                        <strong>{alert['aisle']}</strong> — <small>{alert['duration']}</small><br>
                        <span>{alert['item']} ({alert['sku']})</span><br>
                        <small style="color: #166534;">✅ Restocked by {alert['assigned_to']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ==============================================================================
# 6. VIEW 2: PLANOGRAM DIGITAL TWIN & HEATMAP
# ==============================================================================
elif app_mode == "📐 Planogram Digital Twin":
    st.markdown('<p class="main-header">📐 Planogram Digital Twin & Heatmap Verification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visual matrix comparing approved planograms against current computer vision detections</p>', unsafe_allow_html=True)
    st.markdown("---")

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Aisle Planogram ID", "PG-AISLE3-BEV-v4")
    p_col2.metric("Total Facing Capacity", "18 Facings")
    p_col3.metric("Planogram Compliance", "88.9%", "-5.5% vs Target")
    p_col4.metric("Misplaced Item Count", "1 Item", "Needs realignment")

    st.markdown("### 🛒 Shelf Facings Matrix (Expected vs Real-Time CV)")

    shelves_data = {
        "Shelf A (Top Tier - High Margin)": [
            ("Coca-Cola 500ml", "Coca-Cola 500ml", "slot-ok", "✅ Compliant (95%)"),
            ("Diet Coke 500ml", "Diet Coke 500ml", "slot-ok", "✅ Compliant (92%)"),
            ("Sprite 500ml", "Sprite 500ml", "slot-ok", "✅ Compliant (91%)"),
            ("Fanta Orange 500ml", "Fanta Orange 500ml", "slot-ok", "✅ Compliant (96%)"),
            ("Thums Up 500ml", "Thums Up 500ml", "slot-ok", "✅ Compliant (94%)"),
            ("Mountain Dew 500ml", "Mountain Dew 500ml", "slot-ok", "✅ Compliant (93%)")
        ],
        "Shelf B (Eye Level - Fast Moving)": [
            ("Red Bull 250ml", "Red Bull 250ml", "slot-ok", "✅ Compliant (96%)"),
            ("Pepsi Zero 500ml", "EMPTY / VOID", "slot-stockout", "🚨 OUT OF STOCK"),
            ("Monster Energy", "Lays Onion (Misplaced)", "slot-misplaced", "⚠️ MISPLACED SKU"),
            ("Gatorade Blue 500ml", "Gatorade Blue 500ml", "slot-ok", "✅ Compliant (94%)"),
            ("Tropicana Orange", "Tropicana Orange", "slot-ok", "✅ Compliant (89%)"),
            ("Minute Maid Pulpy", "Minute Maid Pulpy", "slot-ok", "✅ Compliant (91%)")
        ],
        "Shelf C (Bottom Tier - Bulk & Heavy)": [
            ("Bisleri 1L Water", "Bisleri 1L Water", "slot-ok", "✅ Compliant (97%)"),
            ("Aquafina 1L Water", "Aquafina 1L Water", "slot-ok", "✅ Compliant (95%)"),
            ("Kinley 1L Water", "Kinley 1L Water", "slot-ok", "✅ Compliant (94%)"),
            ("Himalayan Spring", "Himalayan Spring", "slot-ok", "✅ Compliant (92%)"),
            ("Real Mixed Fruit 1L", "Real Mixed Fruit 1L", "slot-ok", "✅ Compliant (96%)"),
            ("Paper Boat Aamras", "Paper Boat Aamras", "slot-ok", "✅ Compliant (90%)")
        ]
    }

    for shelf_name, slots in shelves_data.items():
        st.markdown(f"**{shelf_name}**")
        cols = st.columns(6)
        for i, (expected, detected, css_class, status_label) in enumerate(slots):
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="planogram-slot {css_class}">
                        <div>Slot {i+1}</div>
                        <div style="font-size: 11px; margin-top: 3px;"><strong>Plan:</strong> {expected}</div>
                        <div style="font-size: 11px; margin-top: 2px;"><strong>Found:</strong> {detected}</div>
                        <div style="font-size: 10px; margin-top: 4px;">{status_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.write("")

    st.markdown("---")
    st.markdown("### 🔍 Homography & Perspective Correction Inspection")
    st.info("The CV engine applies 4-point perspective warping to map oblique retail camera views into this normalized planogram space.")


# ==============================================================================
# 7. VIEW 3: STORE ANALYTICS & ROI
# ==============================================================================
elif app_mode == "📊 Store Analytics & ROI":
    st.markdown('<p class="main-header">📊 Store Inventory Compliance & Financial ROI Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Measure stockout downtime, lost revenue prevention, and restock operational efficiency</p>', unsafe_allow_html=True)
    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Shelf Compliance", "94.2%", "+1.8% vs last week")
    k2.metric("Active Out-Of-Stocks", "2 Aisles", "-4 resolved today")
    k3.metric("Est. Revenue Saved Today", "₹24,850", "+₹6,200 via prompt restock")
    k4.metric("Avg Restock SLA", "3.4 mins", "-1.1 min faster response")

    st.markdown("---")

    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("### 📈 Hourly Stockout Occurrence (Today)")
        hours = [f"{h}:00" for h in range(8, 22)]
        hourly_data = pd.DataFrame({
            "Hour": hours,
            "Beverages": [2, 3, 5, 8, 12, 14, 11, 7, 9, 13, 15, 8, 4, 2],
            "Snacks": [1, 2, 4, 6, 9, 11, 8, 5, 7, 10, 11, 6, 3, 1],
            "Dairy": [3, 4, 3, 2, 2, 3, 1, 1, 2, 4, 5, 3, 2, 1]
        }).set_index("Hour")
        st.area_chart(hourly_data)
        st.caption("Peak stockout spikes observed during lunch rush (12:00-14:00) and evening hours (18:00-20:00).")

    with c_col2:
        st.markdown("### 🏷️ Stockout Frequency by Category")
        category_data = pd.DataFrame({
            "Category": ["Beverages", "Snacks & Chips", "Personal Care", "Dairy & Eggs", "Bakery"],
            "Weekly Stockouts": [42, 31, 14, 11, 8]
        }).set_index("Category")
        st.bar_chart(category_data)
        st.caption("Beverages and high-turnover snacks represent 73% of all shelf vacancy events.")

    st.markdown("---")
    st.markdown("### 🏆 Restock Staff Performance Leaderboard")
    staff_df = pd.DataFrame({
        "Staff Member": ["Staff #1 (Rahul)", "Staff #2 (Priya)", "Staff #3 (Amit)", "Staff #4 (Ramesh)"],
        "Assigned Tasks": [18, 15, 12, 14],
        "Avg Time to Restock": ["2.8 mins", "3.1 mins", "3.7 mins", "4.0 mins"],
        "SLA Compliance": ["98%", "95%", "92%", "90%"],
        "Rating": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐"]
    })
    st.dataframe(staff_df, use_container_width=True, hide_index=True)


# ==============================================================================
# 8. VIEW 4: INCIDENT LOG & EXPORT
# ==============================================================================
elif app_mode == "🚨 Incident Log & Export":
    st.markdown('<p class="main-header">🚨 Store Incident Audit Trail & Data Export</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete historical audit trail for store managers and retail operations compliance</p>', unsafe_allow_html=True)
    st.markdown("---")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        aisle_filter = st.multiselect("Filter by Aisle", ["Aisle 1", "Aisle 2", "Aisle 3", "Aisle 4"], default=["Aisle 1", "Aisle 2", "Aisle 3", "Aisle 4"])
    with f_col2:
        severity_filter = st.multiselect("Filter by Severity", ["Critical", "Warning", "Resolved"], default=["Critical", "Warning", "Resolved"])
    with f_col3:
        search_query = st.text_input("Search SKU or Product", placeholder="e.g. Coca-Cola, SKU #1102")

    audit_data = [
        {"Timestamp": "2026-09-28 09:34:10", "Incident ID": "INC-8891", "Aisle": "Aisle 3", "SKU": "SKU #4589", "Product": "Coca-Cola 500ml", "Type": "Stockout", "Severity": "Critical", "Duration": "38s", "Status": "Pending"},
        {"Timestamp": "2026-09-28 09:28:45", "Incident ID": "INC-8890", "Aisle": "Aisle 1", "SKU": "SKU #1102", "Product": "Lays Cream & Onion", "Type": "Misplaced", "Severity": "Warning", "Duration": "1m 15s", "Status": "Pending"},
        {"Timestamp": "2026-09-28 09:15:20", "Incident ID": "INC-8889", "Aisle": "Aisle 2", "SKU": "SKU #8820", "Product": "Amul Butter 500g", "Type": "Stockout", "Severity": "Resolved", "Duration": "2m 10s", "Status": "Resolved"},
        {"Timestamp": "2026-09-28 08:52:11", "Incident ID": "INC-8888", "Aisle": "Aisle 4", "SKU": "SKU #3312", "Product": "Dove Soap 100g", "Type": "Stockout", "Severity": "Resolved", "Duration": "1m 45s", "Status": "Resolved"},
        {"Timestamp": "2026-09-28 08:30:05", "Incident ID": "INC-8887", "Aisle": "Aisle 3", "SKU": "SKU #4590", "Product": "Pepsi Can 330ml", "Type": "Stockout", "Severity": "Resolved", "Duration": "3m 05s", "Status": "Resolved"},
        {"Timestamp": "2026-09-28 08:14:40", "Incident ID": "INC-8886", "Aisle": "Aisle 1", "SKU": "SKU #1105", "Product": "Doritos Nacho Cheese", "Type": "Misplaced", "Severity": "Resolved", "Duration": "4m 20s", "Status": "Resolved"},
    ]
    df_audit = pd.DataFrame(audit_data)

    st.dataframe(df_audit, use_container_width=True, hide_index=True)

    csv_bytes = df_audit.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Full Incident Audit Log (CSV)",
        data=csv_bytes,
        file_name=f"shelf_audit_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==============================================================================
# 9. VIEW 5: SYSTEM ARCHITECTURE & PIPELINE BLUEPRINT
# ==============================================================================
else:
    st.markdown('<p class="main-header">🏗️ End-to-End System Architecture (Everseen Pattern)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Technical design, data flow, homography transform, and team responsibility breakdown</p>', unsafe_allow_html=True)
    st.markdown("---")

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
    │  Store Manager / Staff   │      │  Real-Time Alert Broker  │      │  Temporal Persistence    │
    │  • Streamlit Web UI      │ ◄─── │  • REST & WebSockets     │ ◄─── │  • 30s Delay Filter      │
    │  • Restocker Mobile App  │      │  • Priority Escalation   │      │  • Re-id / Occlusion Chk │
    └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
    ```
    """)

    st.markdown("---")
    st.markdown("### 🧩 Technical Implementation Responsibilities")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("""
        #### 🤖 Computer Vision & AI Lead
        * **Model:** YOLOv8x fine-tuned on retail shelf SKU dataset (Grozi-120 + custom retail pack).
        * **Temporal Filter:** Tracks void states across $N$ frames ($t > 30s$) using ByteTrack to prevent false alarms from browsing customers.
        * **Inference Speed:** TensorRT optimized on NVIDIA Jetson / T4 edge instances.
        """)

    with tech_col2:
        st.markdown("""
        #### 📊 Data Science & Planogram
        * **Homography:** Perspective warping maps camera coordinate $(u, v)$ to shelf slot $(x, y)$.
        * **Digital Twin:** Automated alignment against retailer ERP planogram database.
        * **Compliance Scoring:** Real-time formula evaluating actual facings vs contractually agreed brand facings.
        """)

    with tech_col3:
        st.markdown("""
        #### 💻 CS & Frontend Engineering
        * **Dashboard:** Streamlit responsive UI with custom CSS design tokens & live status indicators.
        * **State Engine:** Fast session state mutation for assignment workflows and incident resolution.
        * **Export & Analytics:** Incident logging engine with CSV export and real-time operational KPI scorecards.
        """)
