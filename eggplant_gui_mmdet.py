import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget
)

from mmdet.apis import init_detector, inference_detector


# =========================================================
# SETTINGS
# =========================================================
INPUT_SIZE = 640
CAMERA_INDEX = 0

APP_BG = "#F3EAF7"
CARD_BG = "#E9D9F1"
SIDEBAR_BG = "#8A2BE2"
BTN_BG = "#A020F0"

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SCORE_THR = 0.35

CLASS_INFESTED = "infested"
CLASS_NON_INFESTED = "non-infested"

MMDET_WORKDIR = Path(r"C:\Users\gelob\Desktop\mmdetection\work_dirs")

MODEL_CONFIGS = {
    "SOLOv2": {
        "config": MMDET_WORKDIR / "solov2_eggplant" / "solov2_eggplant.py",
        "checkpoint": MMDET_WORKDIR / "solov2_eggplant" / "best_coco_segm_mAP_epoch_14.pth",
    },
    "RTMDet-Ins": {
        "config": MMDET_WORKDIR / "rtmdet_ins_eggplant" / "rtmdet_ins_eggplant.py",
        "checkpoint": MMDET_WORKDIR / "rtmdet_ins_eggplant" / "best_coco_segm_mAP_epoch_23.pth",
    },
}


# =========================================================
# HELPERS
# =========================================================
def resize_to_fixed_640(img):
    return cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


def bgr_to_pixmap(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def ensure_uint8_mask(mask):
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 1
    return mask


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def union_masks(masks, shape):
    union = np.zeros(shape[:2], dtype=np.uint8)
    for m in masks:
        if m is not None:
            union = np.maximum(union, ensure_uint8_mask(m))
    return union


def overlay_mask(img, mask, color=(0, 0, 255), alpha=0.35):
    out = img.copy()
    if mask is None:
        return out
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.uint8)
    out[mask_bool] = ((1 - alpha) * out[mask_bool] + alpha * color_arr).astype(np.uint8)
    return out


def draw_label_box(img, text, x, y, bg_color=(90, 25, 90), text_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = max(5, x)
    y = max(th + 10, y)

    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 2), font, scale, text_color, thick, cv2.LINE_AA)


def connected_component_boxes(mask, min_area=12):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((x, y, x + w, y + h, area))
    return boxes


def region_name_from_bbox(bbox, image_shape):
    if bbox is None:
        return "-"
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cx < w / 3:
        horiz = "Left"
    elif cx < 2 * w / 3:
        horiz = "Center"
    else:
        horiz = "Right"

    if cy < h / 3:
        vert = "Upper"
    elif cy < 2 * h / 3:
        vert = "Middle"
    else:
        vert = "Lower"

    return f"{vert}-{horiz}"


def build_roi_from_bbox(bbox, image_shape, pad_ratio=1.2):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)

    rx1 = max(0, x1 - px)
    ry1 = max(0, y1 - py)
    rx2 = min(w - 1, x2 + px)
    ry2 = min(h - 1, y2 + py)

    return rx1, ry1, rx2, ry2


def estimate_whole_eggplant_mask(img, seed_mask):
    """
    Estimate the whole eggplant mask using GrabCut guided by the infested region.
    """
    h, w = img.shape[:2]
    bbox = mask_to_bbox(seed_mask)
    if bbox is None:
        return np.zeros((h, w), dtype=np.uint8)

    rx1, ry1, rx2, ry2 = build_roi_from_bbox(bbox, img.shape, pad_ratio=1.2)
    rect_w = max(1, rx2 - rx1)
    rect_h = max(1, ry2 - ry1)

    mask_gc = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        rect = (rx1, ry1, rect_w, rect_h)
        cv2.grabCut(img, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        fg = np.where((mask_gc == 1) | (mask_gc == 3), 1, 0).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        if num_labels <= 1:
            whole_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            whole_mask = (labels == largest_idx).astype(np.uint8)

        # keep seed inside
        whole_mask = np.maximum(whole_mask, seed_mask.astype(np.uint8))

        # if estimate is too tiny, fallback to expanded ROI
        if whole_mask.sum() < seed_mask.sum() * 1.5:
            fallback = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(fallback, (rx1, ry1), (rx2, ry2), 1, -1)
            whole_mask = np.maximum(fallback, seed_mask.astype(np.uint8))

        return whole_mask

    except Exception:
        fallback = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(fallback, (rx1, ry1), (rx2, ry2), 1, -1)
        fallback = np.maximum(fallback, seed_mask.astype(np.uint8))
        return fallback


def compute_severity_from_masks(infested_union, eggplant_union, image_shape):
    """
    Final logic:
    - Severe if most of the eggplant is covered
    - Mild if combined infested area is very small
    - Moderate otherwise
    """
    h, w = image_shape[:2]
    image_area = h * w

    infested_pixels = int(infested_union.sum())
    eggplant_pixels = int(eggplant_union.sum())

    image_pct = (infested_pixels / image_area) * 100.0 if image_area > 0 else 0.0
    eggplant_coverage_pct = (infested_pixels / eggplant_pixels) * 100.0 if eggplant_pixels > 0 else 0.0

    if eggplant_coverage_pct >= 78.0:
        severity = "Severe"
    elif image_pct < 1.5:
        severity = "Mild"
    else:
        severity = "Moderate"

    return severity, image_pct, eggplant_coverage_pct


# =========================================================
# MODEL WRAPPER
# =========================================================
class MMDetSegWrapper:
    def __init__(self, config_path, checkpoint_path, device=DEFAULT_DEVICE):
        self.model = init_detector(str(config_path), str(checkpoint_path), device=device)
        self.classes = [c.strip().lower() for c in self.model.dataset_meta["classes"]]

    def infer(self, image):
        result = inference_detector(self.model, image)
        pred = result.pred_instances

        detections = []
        if pred is None or len(pred) == 0:
            return detections

        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()
        masks = pred.masks.detach().cpu().numpy() if hasattr(pred, "masks") else None

        keep = scores >= SCORE_THR
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if masks is not None:
            masks = masks[keep]

        for i in range(len(bboxes)):
            label_name = self.classes[int(labels[i])]
            mask_i = None
            if masks is not None and i < len(masks):
                mask_i = ensure_uint8_mask(masks[i])

            detections.append({
                "label": label_name,
                "score": float(scores[i]),
                "bbox": bboxes[i].astype(int).tolist(),
                "mask": mask_i,
            })

        return detections


# =========================================================
# MAIN GUI
# =========================================================
class EggplantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eggplant Inspection System - MMDetection")
        self.resize(1150, 720)

        self.original_image = None
        self.current_display_image = None
        self.current_model_name = "RTMDet-Ins"
        self.loaded_models = {}

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setFixedWidth(100)
        sidebar.setObjectName("sidebar")

        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 20, 12, 20)
        side_layout.setSpacing(20)

        self.home_btn = QPushButton("HOME")
        self.results_btn = QPushButton("RESULTS")
        self.about_btn = QPushButton("ABOUT")

        for btn in [self.home_btn, self.results_btn, self.about_btn]:
            btn.setMinimumHeight(60)
            side_layout.addWidget(btn)

        side_layout.addStretch()

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(16)

        top_bar = QHBoxLayout()

        title = QLabel("Results")
        title.setObjectName("titleLabel")

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_CONFIGS.keys()))
        self.model_combo.setCurrentText(self.current_model_name)
        self.model_combo.currentTextChanged.connect(self.change_model)
        self.model_combo.setFixedWidth(220)

        self.upload_btn = QPushButton("Upload Image")
        self.camera_btn = QPushButton("Use Camera")
        self.run_btn = QPushButton("Run Inspection")

        self.upload_btn.clicked.connect(self.upload_image)
        self.camera_btn.clicked.connect(self.capture_camera)
        self.run_btn.clicked.connect(self.run_inspection)

        top_bar.addWidget(title)
        top_bar.addStretch()
        top_bar.addWidget(QLabel("Model:"))
        top_bar.addWidget(self.model_combo)
        top_bar.addWidget(self.upload_btn)
        top_bar.addWidget(self.camera_btn)
        top_bar.addWidget(self.run_btn)

        image_card = QFrame()
        image_card.setObjectName("card")
        image_layout = QVBoxLayout(image_card)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(360)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("background: white; border-radius: 14px;")
        image_layout.addWidget(self.image_label)

        result_card = QFrame()
        result_card.setObjectName("card")
        grid = QGridLayout(result_card)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)

        self.status_chip = QLabel("No Result")
        self.status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_chip.setMinimumHeight(42)
        self.status_chip.setFixedWidth(220)

        self.severity_value = QLabel("-")
        self.cut_region_value = QLabel("-")
        self.conf_value = QLabel("-")
        self.area_pct_value = QLabel("-")

        for w in [self.severity_value, self.cut_region_value, self.conf_value, self.area_pct_value]:
            w.setObjectName("valueBox")
            w.setMinimumHeight(38)

        grid.addWidget(self.status_chip, 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("Severity"), 1, 0)
        grid.addWidget(self.severity_value, 1, 1)
        grid.addWidget(QLabel("Cutting Region"), 2, 0)
        grid.addWidget(self.cut_region_value, 2, 1)
        grid.addWidget(QLabel("Confidence"), 3, 0)
        grid.addWidget(self.conf_value, 3, 1)
        grid.addWidget(QLabel("Infested Area (%)"), 4, 0)
        grid.addWidget(self.area_pct_value, 4, 1)

        content_layout.addLayout(top_bar)
        content_layout.addWidget(image_card, 1)
        content_layout.addWidget(result_card)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(content, 1)

        self.reset_outputs()

    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {APP_BG};
            }}
            #sidebar {{
                background: {SIDEBAR_BG};
                border-top-right-radius: 18px;
                border-bottom-right-radius: 18px;
            }}
            QPushButton {{
                background: {BTN_BG};
                color: white;
                border: none;
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #8E24AA;
            }}
            #card {{
                background: {CARD_BG};
                border-radius: 18px;
                padding: 12px;
            }}
            #titleLabel {{
                font-size: 28px;
                font-weight: 700;
                color: #222222;
            }}
            QLabel {{
                font-size: 16px;
                color: #222222;
            }}
            QComboBox {{
                background: white;
                border-radius: 10px;
                padding: 8px;
                font-size: 14px;
            }}
            #valueBox {{
                background: white;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 15px;
            }}
        """)

    def change_model(self, name):
        self.current_model_name = name

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return

        img = resize_to_fixed_640(img)
        self.original_image = img.copy()
        self.current_display_image = img.copy()
        self.show_image(self.current_display_image)
        self.reset_outputs()

    def capture_camera(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            QMessageBox.warning(self, "Camera Error", "Could not open camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            QMessageBox.warning(self, "Camera Error", "Failed to capture frame.")
            return

        frame = resize_to_fixed_640(frame)

        self.original_image = frame.copy()
        self.current_display_image = frame.copy()
        self.show_image(self.current_display_image)
        self.reset_outputs()

    def show_image(self, img):
        pix = bgr_to_pixmap(img)
        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_display_image is not None:
            self.show_image(self.current_display_image)

    def reset_outputs(self):
        self.status_chip.setText("No Result")
        self.status_chip.setStyleSheet("""
            background:#9E9E9E;
            color:white;
            border-radius:14px;
            font-size:18px;
            font-weight:700;
            padding:8px 14px;
        """)
        self.severity_value.setText("-")
        self.cut_region_value.setText("-")
        self.conf_value.setText("-")
        self.area_pct_value.setText("-")

    def get_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        cfg = MODEL_CONFIGS[model_name]

        if not cfg["config"].exists():
            raise FileNotFoundError(f"Config file not found:\n{cfg['config']}")
        if not cfg["checkpoint"].exists():
            raise FileNotFoundError(f"Checkpoint file not found:\n{cfg['checkpoint']}")

        model = MMDetSegWrapper(cfg["config"], cfg["checkpoint"], DEFAULT_DEVICE)
        self.loaded_models[model_name] = model
        return model

    def run_inspection(self):
        if self.original_image is None:
            QMessageBox.information(self, "No Image", "Please upload or capture an image first.")
            return

        try:
            model = self.get_model(self.current_model_name)
            detections = model.infer(self.original_image.copy())
            vis, summary = self.analyze_and_visualize(self.original_image.copy(), detections)
            self.current_display_image = vis
            self.show_image(vis)
            self.update_result_ui(summary)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def analyze_and_visualize(self, img, detections):
        summary = {
            "status": "NON-INFESTED",
            "severity": "None",
            "cut_region": "Not needed",
            "confidence": "-",
            "area_pct": "0.00%",
        }

        if not detections:
            return img, summary

        filtered = [d for d in detections if d["label"] in [CLASS_INFESTED, CLASS_NON_INFESTED]]
        if not filtered:
            return img, summary

        infested_dets = [d for d in filtered if d["label"] == CLASS_INFESTED]
        clean_dets = [d for d in filtered if d["label"] == CLASS_NON_INFESTED]

        best_inf = max(infested_dets, key=lambda x: x["score"]) if infested_dets else None
        best_clean = max(clean_dets, key=lambda x: x["score"]) if clean_dets else None

        if best_inf is None and best_clean is not None:
            chosen = best_clean
            chosen_status = "NON-INFESTED"
        elif best_clean is None and best_inf is not None:
            chosen = best_inf
            chosen_status = "INFESTED"
        elif best_inf is not None and best_clean is not None:
            if best_inf["score"] >= best_clean["score"]:
                chosen = best_inf
                chosen_status = "INFESTED"
            else:
                chosen = best_clean
                chosen_status = "NON-INFESTED"
        else:
            return img, summary

        for d in clean_dets:
            if d["mask"] is not None:
                img = overlay_mask(img, d["mask"], color=(80, 220, 120), alpha=0.15)

        infested_masks = []
        for d in infested_dets:
            if d["mask"] is not None:
                infested_masks.append(d["mask"])
                img = overlay_mask(img, d["mask"], color=(0, 0, 255), alpha=0.35)

        if chosen_status == "NON-INFESTED":
            x1, y1, x2, y2 = map(int, chosen["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 3)
            draw_label_box(img, f"non-infested {chosen['score']*100:.1f}%", x1, y1)

            summary["status"] = "NON-INFESTED"
            summary["severity"] = "None"
            summary["cut_region"] = "Not needed"
            summary["confidence"] = f"{chosen['score']*100:.1f}%"
            summary["area_pct"] = "0.00%"
            return img, summary

        # Combine all infested masks
        infested_union = union_masks(infested_masks, img.shape)
        if infested_union.sum() == 0 and best_inf is not None:
            x1, y1, x2, y2 = map(int, best_inf["bbox"])
            fallback_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.rectangle(fallback_mask, (x1, y1), (x2, y2), 1, -1)
            infested_union = fallback_mask

        # Estimate whole eggplant mask separately
        eggplant_union = estimate_whole_eggplant_mask(self.original_image.copy(), infested_union)

        severity, image_pct, eggplant_coverage_pct = compute_severity_from_masks(
            infested_union, eggplant_union, img.shape
        )

        comp_boxes = connected_component_boxes(infested_union, min_area=12)
        for x1, y1, x2, y2, _ in comp_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)

        inf_bbox = mask_to_bbox(infested_union)
        egg_bbox = mask_to_bbox(eggplant_union)

        if egg_bbox is not None:
            ex1, ey1, ex2, ey2 = egg_bbox
            cv2.rectangle(img, (ex1, ey1), (ex2, ey2), (120, 255, 120), 2)

        if inf_bbox is not None:
            x1, y1, x2, y2 = inf_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            draw_label_box(img, f"infested {chosen['score']*100:.1f}%", x1, y1)

        if severity == "Severe":
            cut_region = "Not recommended (widespread)"
        else:
            cut_region = region_name_from_bbox(inf_bbox, img.shape)

        cv2.rectangle(img, (0, 0), (img.shape[1], 56), (240, 232, 248), -1)
        cv2.putText(
            img,
            f"{severity} | {chosen['score']*100:.1f}% | Area {image_pct:.2f}% | Cover {eggplant_coverage_pct:.1f}%",
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (70, 20, 120),
            2,
            cv2.LINE_AA
        )

        summary["status"] = "INFESTED"
        summary["severity"] = severity
        summary["cut_region"] = cut_region
        summary["confidence"] = f"{chosen['score']*100:.1f}%"
        summary["area_pct"] = f"{image_pct:.2f}%"

        return img, summary

    def update_result_ui(self, summary):
        status = summary["status"]

        if status == "INFESTED":
            self.status_chip.setText("Infested")
            self.status_chip.setStyleSheet("""
                background:#FF4B4B;
                color:white;
                border-radius:14px;
                font-size:18px;
                font-weight:700;
                padding:8px 14px;
            """)
        else:
            self.status_chip.setText("Non-Infested")
            self.status_chip.setStyleSheet("""
                background:#22AA55;
                color:white;
                border-radius:14px;
                font-size:18px;
                font-weight:700;
                padding:8px 14px;
            """)

        self.severity_value.setText(summary["severity"])
        self.cut_region_value.setText(summary["cut_region"])
        self.conf_value.setText(summary["confidence"])
        self.area_pct_value.setText(summary["area_pct"])


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EggplantGUI()
    window.show()
    sys.exit(app.exec())