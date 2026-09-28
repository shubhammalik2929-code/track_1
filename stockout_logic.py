"""
================================================================================
Everseen x IIT Bombay - Computer Vision Retail Shelf Auditing Engine
File: stockout_logic.py
Author: AI/ML & Computer Vision Lead
Architecture: Temporal Persistence Void Tracker & Planogram Compliance Engine
================================================================================
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union


class StockoutTracker:
    """
    Industrial-Grade Shelf Void & Planogram Stockout Tracker.
    
    Implements temporal persistence filtering to distinguish transient shopper
    occlusions (hands picking or inspecting items) from true out-of-stock shelf voids.
    Also handles planogram SKU verification and misplaced product detection.
    """

    def __init__(
        self,
        persistence_limit_sec: float = 30.0,
        transient_threshold_sec: float = 5.0,
        iou_threshold: float = 0.1,
        ios_threshold: float = 0.20,
    ):
        """
        Initializes the stockout tracker.

        :param persistence_limit_sec: Time in seconds an empty slot must persist 
                                      before triggering a confirmed P1 stockout alert (default: 30s).
        :param transient_threshold_sec: Time window in seconds below which an empty slot 
                                        is treated as transient shopper occlusion (default: 5s).
        :param iou_threshold: Minimum Intersection over Union to associate a detection with a slot.
        :param ios_threshold: Minimum Intersection over Slot Area (IoS) to consider a slot occupied.
                              Solves the issue where smaller products don't satisfy large IoU requirements.
        """
        self.persistence_limit = float(persistence_limit_sec)
        self.transient_threshold = float(transient_threshold_sec)
        self.iou_threshold = float(iou_threshold)
        self.ios_threshold = float(ios_threshold)

        # Tracks start timestamp when a slot first becomes empty: {slot_id: float_timestamp}
        self.missing_timers: Dict[str, float] = {}

        # Tracks active confirmed alerts: {slot_id: alert_dict}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}

        # Historical state tracking for audit logs and MTTR (Mean Time to Restock):
        # {slot_id: {"last_state": str, "state_start": float, "history": List[dict]}}
        self.slot_state_history: Dict[str, Dict[str, Any]] = {}

        # Lifetime KPI counters
        self.total_evaluations_run = 0
        self.total_confirmed_stockouts = 0
        self.total_restock_events = 0
        self.total_misplaced_events = 0

    # --------------------------------------------------------------------------
    # Geometric Spatial Overlap Functions
    # --------------------------------------------------------------------------

    @staticmethod
    def _calculate_iou(boxA: Union[List[float], Tuple[float, ...]], boxB: Union[List[float], Tuple[float, ...]]) -> float:
        """
        Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        boxA_area = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
        boxB_area = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))

        union_area = boxA_area + boxB_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _calculate_ios(product_box: Union[List[float], Tuple[float, ...]], slot_box: Union[List[float], Tuple[float, ...]]) -> float:
        """
        Calculates Intersection over Slot Area (IoS):
        IoS = (Product ∩ Slot) / Slot Area
        
        Essential for retail shelves where a single facing of a can or box
        only occupies a fraction of the full slot volume.
        """
        xA = max(product_box[0], slot_box[0])
        yA = max(product_box[1], slot_box[1])
        xB = min(product_box[2], slot_box[2])
        yB = min(product_box[3], slot_box[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        slot_area = max(0.0, (slot_box[2] - slot_box[0])) * max(0.0, (slot_box[3] - slot_box[1]))
        return inter_area / slot_area if slot_area > 0 else 0.0

    @staticmethod
    def _calculate_iop(product_box: Union[List[float], Tuple[float, ...]], slot_box: Union[List[float], Tuple[float, ...]]) -> float:
        """
        Calculates Intersection over Product Area (IoP):
        IoP = (Product ∩ Slot) / Product Area
        
        Measures what fraction of the detected product lies inside the slot boundaries.
        """
        xA = max(product_box[0], slot_box[0])
        yA = max(product_box[1], slot_box[1])
        xB = min(product_box[2], slot_box[2])
        yB = min(product_box[3], slot_box[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        prod_area = max(0.0, (product_box[2] - product_box[0])) * max(0.0, (product_box[3] - product_box[1]))
        return inter_area / prod_area if prod_area > 0 else 0.0

    # --------------------------------------------------------------------------
    # Core Evaluation Engine
    # --------------------------------------------------------------------------

    def evaluate_slots(
        self,
        predefined_slots: Dict[str, Any],
        detected_boxes: List[Any],
        iou_threshold: Optional[float] = None,
        detected_classes: Optional[List[str]] = None,
        detected_confs: Optional[List[float]] = None,
        current_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates predefined shelf grid slots against currently detected product bounding boxes.
        
        Handles:
        1. Multi-product occupancy and spatial overlap (IoU and IoS).
        2. Temporal persistence tracking (filtering hand occlusions < 5s vs confirmed stockout > 30s).
        3. Planogram compliance & Misplaced SKU detection.
        4. Restock recovery transitions and alert auto-clearing.

        :param predefined_slots: Dict of slot IDs and either:
                                 - Simple pixel coordinates: {'slot_1': [x1, y1, x2, y2]}
                                 - Full planogram metadata: {'slot_1': {'bbox': [x1, y1, x2, y2], 'expected_sku': 'Coca-Cola 500ml', 'tier': 'Tier 1'}}
        :param detected_boxes: List of bounding boxes detected by YOLO model -> [[x1, y1, x2, y2], ...]
                               or list of detection dicts -> [{'bbox': [...], 'class': '...', 'conf': 0.9}]
        :param iou_threshold: Optional override for matching IoU threshold.
        :param detected_classes: Optional list of class names corresponding to detected_boxes.
        :param detected_confs: Optional list of confidence scores corresponding to detected_boxes.
        :param current_time: Optional timestamp (epoch seconds) for deterministic replay / testing.
        :return: List of evaluated slot status dictionaries.
        """
        self.total_evaluations_run += 1
        now = current_time if current_time is not None else time.time()
        effective_iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        # Standardize detected boxes format into unified list of dicts
        normalized_detections = self._normalize_detections(detected_boxes, detected_classes, detected_confs)

        evaluation_results: List[Dict[str, Any]] = []

        for slot_id, slot_info in predefined_slots.items():
            # Parse slot bounding box and planogram metadata
            slot_box, expected_sku, tier_name, extra_meta = self._parse_slot_info(slot_info)

            # Match detected products against this specific slot
            matching_products = []
            best_ios = 0.0
            best_iou = 0.0
            total_ios_coverage = 0.0

            for det in normalized_detections:
                p_box = det["bbox"]
                iou = self._calculate_iou(p_box, slot_box)
                ios = self._calculate_ios(p_box, slot_box)
                iop = self._calculate_iop(p_box, slot_box)

                # Consider matched if either IoU meets threshold OR significant portion of product is in slot
                if iou >= effective_iou or ios >= self.ios_threshold or iop >= 0.40:
                    matching_products.append({
                        "bbox": p_box,
                        "class_name": det.get("class_name", "Product"),
                        "confidence": det.get("confidence", 1.0),
                        "iou": round(iou, 3),
                        "ios": round(ios, 3),
                        "iop": round(iop, 3)
                    })
                    total_ios_coverage += ios
                    if ios > best_ios:
                        best_ios = ios
                    if iou > best_iou:
                        best_iou = iou

            is_occupied = len(matching_products) > 0 and (best_ios >= self.ios_threshold or best_iou >= effective_iou)

            # Evaluate Slot Temporal State
            if is_occupied:
                slot_eval = self._process_occupied_slot(
                    slot_id=slot_id,
                    slot_box=slot_box,
                    expected_sku=expected_sku,
                    tier_name=tier_name,
                    matching_products=matching_products,
                    best_ios=best_ios,
                    best_iou=best_iou,
                    now=now
                )
            else:
                slot_eval = self._process_empty_slot(
                    slot_id=slot_id,
                    slot_box=slot_box,
                    expected_sku=expected_sku,
                    tier_name=tier_name,
                    now=now
                )

            # Merge any extra metadata originally provided
            if extra_meta:
                slot_eval["metadata"] = extra_meta

            evaluation_results.append(slot_eval)

        return evaluation_results

    # --------------------------------------------------------------------------
    # State Processing Internals
    # --------------------------------------------------------------------------

    def _process_occupied_slot(
        self,
        slot_id: str,
        slot_box: List[float],
        expected_sku: Optional[str],
        tier_name: str,
        matching_products: List[Dict[str, Any]],
        best_ios: float,
        best_iou: float,
        now: float
    ) -> Dict[str, Any]:
        """
        Handles state transitions when product presence is detected.
        Checks for SKU compliance and handles restock recoveries.
        """
        # Determine dominant detected product class
        dominant_detected_sku = matching_products[0]["class_name"]
        primary_confidence = matching_products[0]["confidence"]

        # Check Planogram SKU Compliance
        is_planogram_compliant = True
        sku_mismatch_detected = False

        if expected_sku and expected_sku.lower() not in ["any", "product", "generic"]:
            # Check if expected SKU is among detected products
            matched_sku = any(
                p["class_name"].lower() in expected_sku.lower() or expected_sku.lower() in p["class_name"].lower()
                for p in matching_products
            )
            if not matched_sku:
                is_planogram_compliant = False
                sku_mismatch_detected = True

        # Check if slot was previously in missing/stockout state (Recovery / Restock event)
        was_previously_missing = slot_id in self.missing_timers
        duration_empty = 0.0

        if was_previously_missing:
            duration_empty = round(now - self.missing_timers[slot_id], 1)
            # Remove from missing timers
            del self.missing_timers[slot_id]

            # If there was a confirmed alert, resolve it
            if slot_id in self.active_alerts:
                self.total_restock_events += 1
                resolved_alert = self.active_alerts.pop(slot_id)
                resolved_alert["resolved_at"] = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
                resolved_alert["resolution_reason"] = "Restock Verified by CCTV"
                resolved_alert["mttr_sec"] = duration_empty

        # Determine Final Operational State
        if sku_mismatch_detected:
            state = "MISPLACED_PRODUCT"
            action = "DISPATCH_AUDIT_TASK"
            severity = "warning"
            self.total_misplaced_events += 1
        elif was_previously_missing and duration_empty >= self.persistence_limit:
            state = "RESTOCKED_RECOVERED"
            action = "AUTO_RESOLVE_ALERT"
            severity = "info"
        else:
            state = "COMPLIANT"
            action = "NONE"
            severity = "ok"

        self._record_history(slot_id, state, now)

        return {
            "slot_id": slot_id,
            "tier": tier_name,
            "bbox": slot_box,
            "expected_sku": expected_sku,
            "detected_sku": dominant_detected_sku,
            "occupancy_status": "OCCUPIED",
            "state": state,
            "is_compliant": is_planogram_compliant,
            "alert_required": sku_mismatch_detected,
            "severity": severity,
            "action": action,
            "empty_duration_sec": 0.0,
            "persistence_progress_pct": 0.0,
            "matched_item_count": len(matching_products),
            "best_ios": round(best_ios, 3),
            "best_iou": round(best_iou, 3),
            "confidence": primary_confidence,
            "details": f"Slot occupied by {len(matching_products)} item(s). Dominant SKU: '{dominant_detected_sku}'."
                       + (" [PLANOGRAM BREACH: Expected " + expected_sku + "]" if sku_mismatch_detected else "")
        }

    def _process_empty_slot(
        self,
        slot_id: str,
        slot_box: List[float],
        expected_sku: Optional[str],
        tier_name: str,
        now: float
    ) -> Dict[str, Any]:
        """
        Handles state transitions when slot has zero product occupancy.
        Applies temporal persistence filter (Transient Occlusion -> Evaluating -> Stockout Confirmed).
        """
        # Initialize missing timer if first observation
        if slot_id not in self.missing_timers:
            self.missing_timers[slot_id] = now

        elapsed_empty_sec = round(now - self.missing_timers[slot_id], 1)
        persistence_pct = min(100.0, round((elapsed_empty_sec / self.persistence_limit) * 100.0, 1))

        # Multi-stage State Transition
        if elapsed_empty_sec < self.transient_threshold:
            # 0s - 5s: Transient customer hand reach, browsing or temporary camera occlusion
            state = "TRANSIENT_OCCLUSION"
            severity = "info"
            alert_required = False
            action = "SUPPRESS_ALERT_FILTERING"
            details = f"Empty for {elapsed_empty_sec}s (< {self.transient_threshold}s). Filtering transient customer hand motion."

        elif elapsed_empty_sec < self.persistence_limit:
            # 5s - 30s: Evaluating persistence, empty shelf void observed, awaiting confirmation
            remaining = round(self.persistence_limit - elapsed_empty_sec, 1)
            state = "EVALUATING_PERSISTENCE"
            severity = "warning"
            alert_required = False
            action = "MONITORING_COUNTDOWN"
            details = f"Potential void detected. Persistent for {elapsed_empty_sec}s. Confirmation countdown: {remaining}s remaining."

        else:
            # >= 30s: Confirmed persistent shelf stockout!
            state = "STOCKOUT_CONFIRMED"
            severity = "critical"
            alert_required = True
            action = "DISPATCH_RESTOCK_ALERT"
            details = f"CRITICAL STOCKOUT: Shelf slot empty for {elapsed_empty_sec}s (>= {self.persistence_limit}s threshold). Immediate restock required."

            # Register in active alerts cache if not already logged
            if slot_id not in self.active_alerts:
                self.total_confirmed_stockouts += 1
                self.active_alerts[slot_id] = {
                    "alert_id": f"ALT-{uuid.uuid4().hex[:6].upper()}",
                    "slot_id": slot_id,
                    "tier": tier_name,
                    "expected_sku": expected_sku or "Product",
                    "bbox": slot_box,
                    "confirmed_at": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_sec": elapsed_empty_sec,
                    "status": "Pending Restock",
                    "severity": "critical"
                }
            else:
                # Update existing alert duration
                self.active_alerts[slot_id]["duration_sec"] = elapsed_empty_sec

        self._record_history(slot_id, state, now)

        return {
            "slot_id": slot_id,
            "tier": tier_name,
            "bbox": slot_box,
            "expected_sku": expected_sku,
            "detected_sku": None,
            "occupancy_status": "EMPTY",
            "state": state,
            "is_compliant": False,
            "alert_required": alert_required,
            "severity": severity,
            "action": action,
            "empty_duration_sec": elapsed_empty_sec,
            "persistence_progress_pct": persistence_pct,
            "matched_item_count": 0,
            "best_ios": 0.0,
            "best_iou": 0.0,
            "confidence": 0.0,
            "details": details
        }

    # --------------------------------------------------------------------------
    # Helper & Management Utilities
    # --------------------------------------------------------------------------

    def _normalize_detections(
        self,
        detected_boxes: List[Any],
        detected_classes: Optional[List[str]],
        detected_confs: Optional[List[float]]
    ) -> List[Dict[str, Any]]:
        """
        Normalizes various detection input shapes into:
        [{"bbox": [x1, y1, x2, y2], "class_name": str, "confidence": float}, ...]
        """
        normalized = []
        for idx, item in enumerate(detected_boxes):
            if isinstance(item, dict):
                normalized.append({
                    "bbox": item.get("bbox", [0, 0, 0, 0]),
                    "class_name": item.get("class", item.get("class_name", "Product")),
                    "confidence": float(item.get("conf", item.get("confidence", 1.0)))
                })
            elif isinstance(item, (list, tuple)):
                cls_name = detected_classes[idx] if detected_classes and idx < len(detected_classes) else "Product"
                conf = detected_confs[idx] if detected_confs and idx < len(detected_confs) else 1.0
                normalized.append({
                    "bbox": list(item),
                    "class_name": cls_name,
                    "confidence": float(conf)
                })
        return normalized

    def _parse_slot_info(self, slot_info: Any) -> Tuple[List[float], Optional[str], str, Dict[str, Any]]:
        """
        Extracts bbox, expected_sku, tier, and any extra metadata from slot dict or list.
        """
        if isinstance(slot_info, dict):
            bbox = slot_info.get("bbox", [0, 0, 0, 0])
            expected_sku = slot_info.get("expected_sku", slot_info.get("sku", "Product"))
            tier_name = slot_info.get("tier", "Main Shelf")
            extra = {k: v for k, v in slot_info.items() if k not in ["bbox", "expected_sku", "sku", "tier"]}
            return bbox, expected_sku, tier_name, extra
        elif isinstance(slot_info, (list, tuple)):
            return list(slot_info), "Product", "Main Shelf", {}
        return [0, 0, 0, 0], "Product", "Main Shelf", {}

    def _record_history(self, slot_id: str, state: str, now: float):
        """
        Maintains lightweight chronological state transition log per slot.
        """
        if slot_id not in self.slot_state_history:
            self.slot_state_history[slot_id] = {
                "current_state": state,
                "state_entered": now,
                "history": []
            }
        else:
            curr = self.slot_state_history[slot_id]
            if curr["current_state"] != state:
                curr["history"].append({
                    "from_state": curr["current_state"],
                    "to_state": state,
                    "duration_sec": round(now - curr["state_entered"], 1),
                    "timestamp": datetime.fromtimestamp(now).strftime("%H:%M:%S")
                })
                # Keep last 15 transitions
                if len(curr["history"]) > 15:
                    curr["history"].pop(0)
                curr["current_state"] = state
                curr["state_entered"] = now

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Returns all currently active, unresolved stockout alerts."""
        return list(self.active_alerts.values())

    def resolve_alert(self, slot_id: str, reason: str = "Manual Floor Acknowledgment") -> bool:
        """
        Manually or programmatically clears an active stockout alert for a slot.
        """
        if slot_id in self.active_alerts:
            alert = self.active_alerts.pop(slot_id)
            if slot_id in self.missing_timers:
                del self.missing_timers[slot_id]
            return True
        return False

    def reset(self, slot_id: Optional[str] = None):
        """
        Resets tracking timers. If slot_id is provided, resets only that slot;
        otherwise resets all slots.
        """
        if slot_id:
            self.missing_timers.pop(slot_id, None)
            self.active_alerts.pop(slot_id, None)
        else:
            self.missing_timers.clear()
            self.active_alerts.clear()

    def get_shelf_kpis(self, evaluation_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Computes real-time On-Shelf Availability (OSA%) and void statistics.
        """
        if not evaluation_results:
            return {
                "total_slots": 0,
                "occupied_slots": 0,
                "osa_pct": 100.0,
                "confirmed_stockouts": len(self.active_alerts),
                "evaluating_voids": len(self.missing_timers) - len(self.active_alerts),
                "lifetime_stockouts_triggered": self.total_confirmed_stockouts,
                "lifetime_restocks_verified": self.total_restock_events
            }

        total = len(evaluation_results)
        occupied = sum(1 for s in evaluation_results if s["occupancy_status"] == "OCCUPIED")
        compliant = sum(1 for s in evaluation_results if s["state"] == "COMPLIANT")
        stockouts = sum(1 for s in evaluation_results if s["state"] == "STOCKOUT_CONFIRMED")
        evaluating = sum(1 for s in evaluation_results if s["state"] == "EVALUATING_PERSISTENCE")
        misplaced = sum(1 for s in evaluation_results if s["state"] == "MISPLACED_PRODUCT")

        osa_pct = round((compliant / total) * 100.0, 1) if total > 0 else 100.0

        return {
            "total_slots": total,
            "occupied_slots": occupied,
            "compliant_slots": compliant,
            "osa_pct": osa_pct,
            "confirmed_stockouts": stockouts,
            "evaluating_persistence": evaluating,
            "misplaced_products": misplaced,
            "lifetime_stockouts_triggered": self.total_confirmed_stockouts,
            "lifetime_restocks_verified": self.total_restock_events
        }


# ==============================================================================
# SELF-TEST & SIMULATION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("EVERSEEN x IIT BOMBAY - STOCKOUT TRACKER VERIFICATION PIPELINE")
    print("=" * 75)

    # Initialize tracker with 30s persistence limit and 5s transient filter
    tracker = StockoutTracker(persistence_limit_sec=30.0, transient_threshold_sec=5.0)

    # Define 4 shelf slots with Planogram specifications
    slots = {
        "slot_1": {"bbox": [50, 100, 180, 300], "expected_sku": "Coca-Cola 500ml", "tier": "Shelf A"},
        "slot_2": {"bbox": [200, 100, 330, 300], "expected_sku": "Diet Coke 500ml", "tier": "Shelf A"},
        "slot_3": {"bbox": [350, 100, 480, 300], "expected_sku": "Sprite 500ml", "tier": "Shelf A"},
        "slot_4": {"bbox": [500, 100, 630, 300], "expected_sku": "Fanta Orange 500ml", "tier": "Shelf A"},
    }

    t0 = 1000.0  # Base epoch simulation timestamp

    # --------------------------------------------------------------------------
    # STEP 1: t = 0s -> All slots fully stocked and compliant
    # --------------------------------------------------------------------------
    print("\n[Step 1] t = 0.0s: All products present on shelf...")
    detections_t0 = [
        {"bbox": [55, 110, 175, 295], "class": "Coca-Cola 500ml", "conf": 0.95},
        {"bbox": [205, 110, 325, 295], "class": "Diet Coke 500ml", "conf": 0.94},
        {"bbox": [355, 110, 475, 295], "class": "Sprite 500ml", "conf": 0.92},
        {"bbox": [505, 110, 625, 295], "class": "Fanta Orange 500ml", "conf": 0.96},
    ]
    res_t0 = tracker.evaluate_slots(slots, detections_t0, current_time=t0)
    for r in res_t0:
        print(f"  -> {r['slot_id']} [{r['expected_sku']}]: {r['state']} | Alert: {r['alert_required']}")

    # --------------------------------------------------------------------------
    # STEP 2: t = 3s -> Shopper hand reaches in to grab Diet Coke (slot_2 missing)
    # --------------------------------------------------------------------------
    print("\n[Step 2] t = 3.0s: Customer hand reaches into Slot 2 (Diet Coke)...")
    detections_t1 = [
        {"bbox": [55, 110, 175, 295], "class": "Coca-Cola 500ml", "conf": 0.95},
        # slot_2 temporarily occluded/missing
        {"bbox": [355, 110, 475, 295], "class": "Sprite 500ml", "conf": 0.92},
        {"bbox": [505, 110, 625, 295], "class": "Fanta Orange 500ml", "conf": 0.96},
    ]
    res_t1 = tracker.evaluate_slots(slots, detections_t1, current_time=t0 + 3.0)
    slot_2_r1 = next(r for r in res_t1 if r["slot_id"] == "slot_2")
    print(f"  -> slot_2 State: {slot_2_r1['state']} (Duration: {slot_2_r1['empty_duration_sec']}s)")
    print(f"     Action: {slot_2_r1['action']} (Alert Required: {slot_2_r1['alert_required']})")
    assert slot_2_r1["state"] == "TRANSIENT_OCCLUSION", "Expected transient occlusion"

    # --------------------------------------------------------------------------
    # STEP 3: t = 18s -> Slot 2 remains empty for 18 seconds (Persistence check)
    # --------------------------------------------------------------------------
    print("\n[Step 3] t = 18.0s: Slot 2 remains empty (Evaluating Persistence)...")
    res_t2 = tracker.evaluate_slots(slots, detections_t1, current_time=t0 + 18.0)
    slot_2_r2 = next(r for r in res_t2 if r["slot_id"] == "slot_2")
    print(f"  -> slot_2 State: {slot_2_r2['state']} (Duration: {slot_2_r2['empty_duration_sec']}s, Progress: {slot_2_r2['persistence_progress_pct']}%)")
    print(f"     Action: {slot_2_r2['action']} (Alert Required: {slot_2_r2['alert_required']})")
    assert slot_2_r2["state"] == "EVALUATING_PERSISTENCE", "Expected evaluating persistence"

    # --------------------------------------------------------------------------
    # STEP 4: t = 35s -> Slot 2 empty for 35s (> 30s threshold -> ALERT CONFIRMED)
    # --------------------------------------------------------------------------
    print("\n[Step 4] t = 35.0s: Slot 2 empty exceeds 30s threshold -> STOCKOUT CONFIRMED!")
    res_t3 = tracker.evaluate_slots(slots, detections_t1, current_time=t0 + 35.0)
    slot_2_r3 = next(r for r in res_t3 if r["slot_id"] == "slot_2")
    print(f"  -> slot_2 State: {slot_2_r3['state']} (Duration: {slot_2_r3['empty_duration_sec']}s)")
    print(f"     Action: {slot_2_r3['action']} | Severity: {slot_2_r3['severity']} | Alert Required: {slot_2_r3['alert_required']}")
    assert slot_2_r3["state"] == "STOCKOUT_CONFIRMED", "Expected stockout confirmed"

    active_alerts = tracker.get_active_alerts()
    print(f"  -> Active Alert Queue Count: {len(active_alerts)}")
    print(f"     Dispatched Alert Payload: {active_alerts[0]}")

    # --------------------------------------------------------------------------
    # STEP 5: t = 40s -> Misplaced Item placed in Slot 3 (Mountain Dew in Sprite slot)
    # --------------------------------------------------------------------------
    print("\n[Step 5] t = 40.0s: Misplaced product placed in Slot 3 (Mountain Dew in Sprite slot)...")
    detections_misplaced = [
        {"bbox": [55, 110, 175, 295], "class": "Coca-Cola 500ml", "conf": 0.95},
        # slot_2 still empty
        {"bbox": [355, 110, 475, 295], "class": "Mountain Dew 500ml", "conf": 0.91},  # WRONG SKU!
        {"bbox": [505, 110, 625, 295], "class": "Fanta Orange 500ml", "conf": 0.96},
    ]
    res_t4 = tracker.evaluate_slots(slots, detections_misplaced, current_time=t0 + 40.0)
    slot_3_r4 = next(r for r in res_t4 if r["slot_id"] == "slot_3")
    print(f"  -> slot_3 State: {slot_3_r4['state']} (Expected: {slot_3_r4['expected_sku']}, Detected: {slot_3_r4['detected_sku']})")
    print(f"     Action: {slot_3_r4['action']} | Alert Required: {slot_3_r4['alert_required']}")
    assert slot_3_r4["state"] == "MISPLACED_PRODUCT", "Expected misplaced product detection"

    # --------------------------------------------------------------------------
    # STEP 6: t = 60s -> Restock event: Slot 2 refilled with Diet Coke
    # --------------------------------------------------------------------------
    print("\n[Step 6] t = 60.0s: Associate restocks Slot 2 with Diet Coke...")
    detections_restocked = [
        {"bbox": [55, 110, 175, 295], "class": "Coca-Cola 500ml", "conf": 0.95},
        {"bbox": [205, 110, 325, 295], "class": "Diet Coke 500ml", "conf": 0.94},  # RESTOCKED!
        {"bbox": [355, 110, 475, 295], "class": "Sprite 500ml", "conf": 0.92},
        {"bbox": [505, 110, 625, 295], "class": "Fanta Orange 500ml", "conf": 0.96},
    ]
    res_t5 = tracker.evaluate_slots(slots, detections_restocked, current_time=t0 + 60.0)
    slot_2_r5 = next(r for r in res_t5 if r["slot_id"] == "slot_2")
    print(f"  -> slot_2 State: {slot_2_r5['state']} | Action: {slot_2_r5['action']}")
    print(f"  -> Remaining Active Alerts: {len(tracker.get_active_alerts())}")

    kpis = tracker.get_shelf_kpis(res_t5)
    print("\n" + "=" * 75)
    print("FINAL AUDIT DASHBOARD KPIs:")
    print(f"  On-Shelf Availability (OSA): {kpis['osa_pct']}%")
    print(f"  Compliant Slots:             {kpis['compliant_slots']}/{kpis['total_slots']}")
    print(f"  Active Stockouts:            {kpis['confirmed_stockouts']}")
    print(f"  Verified Restock Recoveries: {kpis['lifetime_restocks_verified']}")
    print("=" * 75)
    print("ALL TESTS PASSED SUCCESSFULLY!")
