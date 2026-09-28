from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import time
from datetime import datetime

# ==========================================
# 1. ADD YOUR CV ENGINE IMPORTS HERE
# ==========================================
from cv_engine.detector import ShelfDetector
from cv_engine.stockout_logic import StockoutTracker

app = FastAPI(
    title="Shelf Auditing & Stockout Backend",
    description="E-Cell IIT Bombay Hackathon - Track 1 Backend API",
    version="1.0.0"
)

# ==========================================
# 2. INITIALIZE THEM GLOBALLY HERE
# ==========================================
detector = ShelfDetector(model_path="yolov8n.pt")
tracker = StockoutTracker(persistence_limit_sec=30)

# (Rest of your FastAPI routes like app.get(), app.post(), etc., follow below...)


"""
================================================================================
Everseen x IIT Bombay - Retail Shelf Auditing AI Backend Engine
File: backend.py
Author: AI/ML & Computer Vision Lead
Architecture: FastAPI v2.0 + Dual CV Engine + Temporal Persistence Tracker
================================================================================
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import base64
import io
import time
from datetime import datetime, timedelta

from detector import ShelfDetector
from stockout_logic import StockoutTracker

# Instantiate unified CV shelf detector & temporal StockoutTracker
global_shelf_detector = ShelfDetector(persistence_threshold_sec=30.0)
global_stockout_tracker = StockoutTracker(persistence_limit_sec=30.0)

# ==============================================================================
# 1. FASTAPI APPLICATION SETUP
# ==============================================================================
app = FastAPI(
    title="Everseen x IIT Bombay - Shelf Auditing CV Engine API",
    description="Edge-accelerated Automated Shelf Auditing, Planogram Compliance, and Stockout Alert Backend.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 2. IN-MEMORY STATE & DATABASE SIMULATION
# ==============================================================================
START_TIME = datetime.now()

CAMERA_FEEDS = {
    "CAM-01": {"id": "CAM-01", "aisle": "Aisle 3 (Beverages & Soda)", "rtsp": "rtsp://edge.store14.internal/ch01", "fps": 30, "status": "ONLINE", "resolution": "1920x1080"},
    "CAM-02": {"id": "CAM-02", "aisle": "Aisle 1 (Snacks & Confectionery)", "rtsp": "rtsp://edge.store14.internal/ch02", "fps": 30, "status": "ONLINE", "resolution": "1920x1080"},
    "CAM-03": {"id": "CAM-03", "aisle": "Aisle 2 (Dairy & Refrigerated)", "rtsp": "rtsp://edge.store14.internal/ch03", "fps": 25, "status": "ONLINE", "resolution": "1920x1080"},
    "CAM-04": {"id": "CAM-04", "aisle": "Aisle 4 (Personal Care & Hygiene)", "rtsp": "rtsp://edge.store14.internal/ch04", "fps": 30, "status": "ONLINE", "resolution": "1920x1080"},
}

ACTIVE_ALERTS = [
    {
        "alert_id": "ALT-101",
        "aisle": "Aisle 3, Shelf B",
        "sku": "SKU #4589",
        "product_name": "Coca-Cola 500ml",
        "category": "Beverages",
        "issue_type": "Critical Stockout",
        "duration_sec": 38,
        "status": "Pending Restock",
        "assigned_to": "Unassigned",
        "severity": "critical",
        "timestamp": (datetime.now() - timedelta(seconds=38)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "alert_id": "ALT-102",
        "aisle": "Aisle 1, Shelf A",
        "sku": "SKU #1102",
        "product_name": "Lays Cream & Onion",
        "category": "Snacks",
        "issue_type": "Misplaced Item",
        "duration_sec": 75,
        "status": "Pending Review",
        "assigned_to": "Unassigned",
        "severity": "warning",
        "timestamp": (datetime.now() - timedelta(seconds=75)).strftime("%Y-%m-%d %H:%M:%S")
    }
]

RESOLVED_ALERTS = [
    {
        "alert_id": "ALT-100",
        "aisle": "Aisle 2, Shelf C",
        "sku": "SKU #8820",
        "product_name": "Amul Butter 500g",
        "category": "Dairy",
        "issue_type": "Restocked",
        "duration_sec": 130,
        "status": "Completed",
        "assigned_to": "Staff #4 (Ramesh)",
        "severity": "success",
        "timestamp": (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    }
]

PLANOGRAM_STORE = {
    "aisle3": {
        "planogram_id": "PG-AISLE3-BEV-v4",
        "aisle_name": "Aisle 3 (Beverages)",
        "compliance_target": 95.0,
        "current_compliance": 88.9,
        "facings": [
            {"tier": "Shelf A", "slot": 1, "expected": "Coca-Cola 500ml", "detected": "Coca-Cola 500ml", "status": "COMPLIANT", "confidence": 0.95},
            {"tier": "Shelf A", "slot": 2, "expected": "Diet Coke 500ml", "detected": "Diet Coke 500ml", "status": "COMPLIANT", "confidence": 0.92},
            {"tier": "Shelf A", "slot": 3, "expected": "Sprite 500ml", "detected": "Sprite 500ml", "status": "COMPLIANT", "confidence": 0.91},
            {"tier": "Shelf A", "slot": 4, "expected": "Fanta Orange 500ml", "detected": "Fanta Orange 500ml", "status": "COMPLIANT", "confidence": 0.96},
            {"tier": "Shelf A", "slot": 5, "expected": "Thums Up 500ml", "detected": "Thums Up 500ml", "status": "COMPLIANT", "confidence": 0.94},
            {"tier": "Shelf A", "slot": 6, "expected": "Mountain Dew 500ml", "detected": "Mountain Dew 500ml", "status": "COMPLIANT", "confidence": 0.93},
            {"tier": "Shelf B", "slot": 1, "expected": "Red Bull 250ml", "detected": "Red Bull 250ml", "status": "COMPLIANT", "confidence": 0.96},
            {"tier": "Shelf B", "slot": 2, "expected": "Pepsi Zero 500ml", "detected": "EMPTY / VOID", "status": "STOCKOUT", "confidence": 0.91},
            {"tier": "Shelf B", "slot": 3, "expected": "Monster Energy", "detected": "Lays Onion (Misplaced)", "status": "MISPLACED", "confidence": 0.88},
            {"tier": "Shelf B", "slot": 4, "expected": "Gatorade Blue 500ml", "detected": "Gatorade Blue 500ml", "status": "COMPLIANT", "confidence": 0.94},
            {"tier": "Shelf B", "slot": 5, "expected": "Tropicana Orange", "detected": "Tropicana Orange", "status": "COMPLIANT", "confidence": 0.89},
            {"tier": "Shelf B", "slot": 6, "expected": "Minute Maid Pulpy", "detected": "Minute Maid Pulpy", "status": "COMPLIANT", "confidence": 0.91},
            {"tier": "Shelf C", "slot": 1, "expected": "Bisleri 1L Water", "detected": "Bisleri 1L Water", "status": "COMPLIANT", "confidence": 0.97},
            {"tier": "Shelf C", "slot": 2, "expected": "Aquafina 1L Water", "detected": "Aquafina 1L Water", "status": "COMPLIANT", "confidence": 0.95},
            {"tier": "Shelf C", "slot": 3, "expected": "Kinley 1L Water", "detected": "Kinley 1L Water", "status": "COMPLIANT", "confidence": 0.94},
            {"tier": "Shelf C", "slot": 4, "expected": "Himalayan Spring", "detected": "Himalayan Spring", "status": "COMPLIANT", "confidence": 0.92},
            {"tier": "Shelf C", "slot": 5, "expected": "Real Mixed Fruit 1L", "detected": "Real Mixed Fruit 1L", "status": "COMPLIANT", "confidence": 0.96},
            {"tier": "Shelf C", "slot": 6, "expected": "Paper Boat Aamras", "detected": "Paper Boat Aamras", "status": "COMPLIANT", "confidence": 0.90}
        ]
    }
}

# ==============================================================================
# 3. PYDANTIC REQUEST / RESPONSE SCHEMAS
# ==============================================================================
class RestockResolution(BaseModel):
    alert_id: str
    resolved_by: str

class AlertSimulationRequest(BaseModel):
    aisle: str
    product_name: str
    sku: str
    category: str = "General"
    severity: str = "critical"

class EvaluateSlotsRequest(BaseModel):
    predefined_slots: Dict[str, Any]
    detected_boxes: List[Any]
    iou_threshold: Optional[float] = 0.1
    detected_classes: Optional[List[str]] = None
    detected_confs: Optional[List[float]] = None

# ==============================================================================
# 4. COMPUTER VISION HELPER (Analytical Shelf Grid Fallback Engine)
# ==============================================================================
def process_cctv_frame_cv(image_np: np.ndarray, confidence_thresh: float = 0.5):
    """
    Simulates real-time retail shelf Computer Vision inference:
    - Segments shelf tiers and contours
    - Detects empty shelf spaces (voids)
    - Returns annotated image with bounding boxes & telemetry
    """
    h, w, c = image_np.shape
    annotated = image_np.copy()
    detections = []

    # Convert to grayscale for contrast & edge analysis
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours representing potential shelf items
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > 30 and bh > 40 and bw < w * 0.8:
            valid_boxes.append((x, y, bw, bh))

    # If not enough contours found, create realistic geometric shelf slots
    if len(valid_boxes) < 4:
        step_x = max(w // 6, 60)
        step_y = max(h // 3, 80)
        for row in range(3):
            for col in range(5):
                bx = int(col * step_x + 20)
                by = int(row * step_y + 30)
                bw = int(step_x * 0.8)
                bh = int(step_y * 0.75)
                if bx + bw < w and by + bh < h:
                    valid_boxes.append((bx, by, bw, bh))

    stockout_found = False
    for i, (bx, by, bw, bh) in enumerate(valid_boxes[:8]):
        # Inject one realistic stockout void
        if i == 2:
            stockout_found = True
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (40, 40, 230), 2)
            cv2.line(annotated, (bx, by), (bx + bw, by + bh), (60, 60, 230), 1)
            cv2.line(annotated, (bx, by + bh), (bx + bw, by), (60, 60, 230), 1)
            cv2.rectangle(annotated, (bx, max(0, by - 20)), (bx + bw, by), (40, 40, 230), -1)
            cv2.putText(annotated, "STOCKOUT VOID", (bx + 4, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            detections.append({
                "label": "STOCKOUT_VOID",
                "confidence": 0.94,
                "bbox": [bx, by, bw, bh],
                "status": "ANOMALY",
                "sku": "SKU #4589"
            })
        elif i == 4:
            # Misplaced item
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (30, 180, 240), 2)
            cv2.putText(annotated, "MISPLACED (88%)", (bx + 2, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 180, 240), 1)
            detections.append({
                "label": "MISPLACED_PRODUCT",
                "confidence": 0.88,
                "bbox": [bx, by, bw, bh],
                "status": "MISPLACED",
                "sku": "SKU #1102"
            })
        else:
            # Compliant SKU
            conf = round(float(np.random.uniform(0.89, 0.97)), 2)
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (46, 204, 113), 2)
            cv2.putText(annotated, f"SKU #{100+i}: {int(conf*100)}%", (bx + 2, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (46, 204, 113), 1)
            detections.append({
                "label": f"SKU #{100+i}",
                "confidence": conf,
                "bbox": [bx, by, bw, bh],
                "status": "COMPLIANT",
                "sku": f"SKU #{100+i}"
            })

    # Add OSD watermark
    cv2.rectangle(annotated, (0, 0), (w, 30), (15, 23, 42), -1)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, f"EDGE-CV INFERENCE | {timestamp_str} | YOLOv8-Retail | Voids: {1 if stockout_found else 0}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 240), 1)

    # Encode annotated frame to JPEG base64 for API transmission
    _, buffer = cv2.imencode('.jpg', annotated)
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return detections, base64_str, stockout_found

# ==============================================================================
# 5. CORE API ENDPOINTS
# ==============================================================================
@app.get("/")
def read_root():
    """Health check and telemetry endpoint."""
    uptime_sec = int((datetime.now() - START_TIME).total_seconds())
    return {
        "status": "Online",
        "service": "Everseen x IIT Bombay Shelf Auditing AI Backend",
        "version": "2.0.0",
        "uptime_seconds": uptime_sec,
        "docs_url": "/docs",
        "endpoints": [
            "/api/v1/cameras",
            "/api/v1/alerts",
            "/api/v1/alerts/simulate",
            "/api/v1/resolve-alert",
            "/api/v1/detect-frame",
            "/api/v1/evaluate-slots",
            "/api/v1/planogram/{aisle_id}",
            "/api/v1/analytics/kpis",
            "/api/v1/incidents/export"
        ]
    }

@app.get("/api/v1/cameras")
def get_camera_streams():
    """Returns all active retail camera feeds and their telemetry status."""
    return {"total_cameras": len(CAMERA_FEEDS), "cameras": list(CAMERA_FEEDS.values())}

@app.get("/api/v1/alerts")
def get_alerts():
    """Returns active and resolved stockout alerts."""
    return {
        "total_active": len(ACTIVE_ALERTS),
        "total_resolved": len(RESOLVED_ALERTS),
        "active_alerts": ACTIVE_ALERTS,
        "resolved_alerts": RESOLVED_ALERTS
    }

@app.post("/api/v1/alerts/simulate")
def simulate_stockout_alert(payload: AlertSimulationRequest):
    """Dynamically simulates an edge-detected stockout alert on the backend."""
    new_alert = {
        "alert_id": f"ALT-{np.random.randint(200, 999)}",
        "aisle": payload.aisle,
        "sku": payload.sku,
        "product_name": payload.product_name,
        "category": payload.category,
        "issue_type": "Critical Stockout",
        "duration_sec": 5,
        "status": "Pending Restock",
        "assigned_to": "Unassigned",
        "severity": payload.severity,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    ACTIVE_ALERTS.insert(0, new_alert)
    return {"status": "success", "alert": new_alert}

@app.post("/api/v1/resolve-alert")
def resolve_alert(resolution: RestockResolution):
    """Marks an active stockout alert as resolved by a designated staff member."""
    global ACTIVE_ALERTS, RESOLVED_ALERTS
    found_alert = None
    remaining_alerts = []

    for alert in ACTIVE_ALERTS:
        if alert["alert_id"] == resolution.alert_id:
            found_alert = alert
        else:
            remaining_alerts.append(alert)

    if not found_alert:
        raise HTTPException(status_code=404, detail="Alert ID not found in active list.")

    ACTIVE_ALERTS = remaining_alerts
    found_alert["status"] = "Completed"
    found_alert["severity"] = "success"
    found_alert["assigned_to"] = resolution.resolved_by
    found_alert["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    RESOLVED_ALERTS.insert(0, found_alert)

    return {
        "status": "success",
        "message": f"Alert {resolution.alert_id} resolved by {resolution.resolved_by}.",
        "alert": found_alert
    }

@app.post("/api/v1/detect-frame")
async def detect_frame(file: UploadFile = File(...), confidence: float = 0.5):
    """
    Accepts raw CCTV shelf image bytes, executes Computer Vision void detection,
    and returns detected SKU bounding boxes + Base64 annotated image.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image encoding.")

        cv_res = global_shelf_detector.detect_shelf(frame, conf_threshold=confidence, annotate=True)
        _, buffer = cv2.imencode('.jpg', cv_res["annotated_frame"])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "filename": file.filename,
            "engine": cv_res["engine"],
            "image_dimensions": {"width": frame.shape[1], "height": frame.shape[0]},
            "total_products": cv_res["total_products"],
            "total_voids": cv_res["total_voids"],
            "confirmed_stockouts": cv_res["confirmed_stockouts"],
            "stockout_detected": cv_res["total_voids"] > 0,
            "products": cv_res["products"],
            "voids": cv_res["voids"],
            "annotated_image_base64": annotated_b64,
            "inference_time_ms": cv_res["inference_time_ms"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/evaluate-slots")
def evaluate_shelf_slots(req: EvaluateSlotsRequest):
    """
    Evaluates shelf grid slots using the enhanced temporal StockoutTracker.
    Distinguishes transient occlusions (<5s) from persistent stockouts (>30s)
    and validates planogram SKU compliance.
    """
    try:
        results = global_stockout_tracker.evaluate_slots(
            predefined_slots=req.predefined_slots,
            detected_boxes=req.detected_boxes,
            iou_threshold=req.iou_threshold,
            detected_classes=req.detected_classes,
            detected_confs=req.detected_confs
        )
        kpis = global_stockout_tracker.get_shelf_kpis(results)
        active_alerts = global_stockout_tracker.get_active_alerts()

        return {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluated_slots": results,
            "kpis": kpis,
            "active_alerts": active_alerts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/planogram/{aisle_id}")
def get_planogram(aisle_id: str):
    """Fetches the digital twin planogram matrix for a specified aisle."""
    if aisle_id not in PLANOGRAM_STORE:
        raise HTTPException(status_code=404, detail="Planogram not found for specified aisle.")
    return PLANOGRAM_STORE[aisle_id]

@app.get("/api/v1/analytics/kpis")
def get_store_kpis():
    """Returns high-level business compliance and inventory financial metrics."""
    return {
        "overall_compliance_pct": 94.2,
        "compliance_delta": "+1.8% vs last week",
        "active_stockout_aisles": len(ACTIVE_ALERTS),
        "stockout_delta": "-4 resolved today",
        "estimated_revenue_saved_inr": 24850,
        "revenue_delta": "+₹6,200 via prompt restock",
        "avg_restock_sla_minutes": 3.4,
        "sla_delta": "-1.1 min faster response",
        "camera_health": "100% (4/4 Online)"
    }

@app.get("/api/v1/incidents/export")
def export_incidents_csv():
    """Generates and streams a CSV of historical incident audit records."""
    all_events = ACTIVE_ALERTS + RESOLVED_ALERTS
    output = io.StringIO()
    output.write("Alert ID,Aisle,Product,SKU,Issue Type,Duration,Status,Assigned To,Timestamp\n")
    for ev in all_events:
        output.write(f"{ev.get('alert_id')},{ev.get('aisle')},{ev.get('product_name')},{ev.get('sku')},{ev.get('issue_type')},{ev.get('duration_sec')}s,{ev.get('status')},{ev.get('assigned_to')},{ev.get('timestamp')}\n")
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=shelf_audit_incidents.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
