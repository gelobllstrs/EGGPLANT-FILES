import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mmdet.apis import init_detector, inference_detector


# =========================================================
# SETTINGS
# =========================================================
INPUT_SIZE = 640
DISPLAY_SIZE = 460
CAMERA_INDEX = 0
SCORE_THR = 0.35

APP_BG = "#F3EAF7"
CARD_BG = "#E9D9F1"
SIDEBAR_BG = "#8A2BE2"
BTN_BG = "#A020F0"

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# EXACT class names and order used in BOTH configs:
# ('full_infested', 'non_infested', 'partial_infested')
CLASS_FULL = "full_infested"
CLASS_NON_INFESTED = "non_infested"
CLASS_PARTIAL = "partial_infested"

MMDET_ROOT = Path(r"C:\Users\Ezekiel\Desktop\mmdetection")
CONFIGS_DIR = MMDET_ROOT / "configs"
WORKDIRS_DIR = MMDET_ROOT / "work_dirs"

MODEL_CONFIGS = {
    "SOLOv2": {
        "config": CONFIGS_DIR / "solov2" / "solov2_eggplant.py",
        "checkpoint": WORKDIRS_DIR / "solov2_eggplant_3class_30e" / "best_coco_segm_mAP_epoch_10.pth",
    },
    "RTMDet-Ins": {
        "config": CONFIGS_DIR / "rtmdet" / "rtmdet_ins_eggplant.py",
        "checkpoint": WORKDIRS_DIR / "rtmdet_ins_l_eggplant_3class_30e" / "best_coco_segm_mAP_epoch_28.pth",
    },
}


# =========================================================
# HELPERS
# =========================================================
def fit_image_to_canvas(img: np.ndarray, canvas_size: int = INPUT_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size.")

    scale = min(canvas_size / w, canvas_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_size, canvas_size, 3), 245, dtype=np.uint8)

    x = (canvas_size - new_w) // 2
    y = (canvas_size - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def bgr_to_pixmap(img_bgr: np.ndarray) -> QPixmap:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def ensure_uint8_mask(mask):
    if mask is None:
        return None
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 1
    return mask


def union_masks(masks, shape):
    union = np.zeros(shape[:2], dtype=np.uint8)
    for m in masks:
        if m is not None:
            union = np.maximum(union, ensure_uint8_mask(m))
    return union


def mask_to_bbox(mask):
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay_mask(img, mask, color=(0, 0, 255), alpha=0.35):
    if mask is None:
        return img.copy()
    out = img.copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.uint8)
    out[mask_bool] = ((1 - alpha) * out[mask_bool] + alpha * color_arr).astype(np.uint8)
    return out


def draw_label_box(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.60
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = max(5, x)
    y = max(th + 12, y)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 12, y + 4), (90, 25, 90), -1)
    cv2.putText(img, text, (x + 6, y - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def connected_component_boxes(mask, min_area=15):
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
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

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


def format_conf(score):
    return f"{score * 100:.1f}%"


# =========================================================
# DECISION LOGIC
# =========================================================
def summarize_detection(img, detections):
    severity_map = {
        "Non-Infested": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
    }

    summary = {
        "status": "Non-Infested",
        "severity": "Non-Infested (0)",
        "cut_region": "Not needed",
    }

    vis = img.copy()

    if not detections:
        return vis, summary

    full_dets = [d for d in detections if d["label"] == CLASS_FULL]
    non_infested_dets = [d for d in detections if d["label"] == CLASS_NON_INFESTED]
    partial_dets = [d for d in detections if d["label"] == CLASS_PARTIAL]

    best_full = max(full_dets, key=lambda d: d["score"], default=None)
    best_non_infested = max(non_infested_dets, key=lambda d: d["score"], default=None)
    best_partial = max(partial_dets, key=lambda d: d["score"], default=None)

    # Priority based on your class meaning, not on class index.
    if best_full is not None and (
        best_non_infested is None or best_full["score"] >= best_non_infested["score"]
    ):
        chosen = best_full
        chosen_class = CLASS_FULL
    elif best_partial is not None and (
        best_non_infested is None or best_partial["score"] >= best_non_infested["score"]
    ):
        chosen = best_partial
        chosen_class = CLASS_PARTIAL
    elif best_non_infested is not None:
        chosen = best_non_infested
        chosen_class = CLASS_NON_INFESTED
    else:
        return vis, summary

    if chosen_class == CLASS_NON_INFESTED:
        if chosen["bbox"] is not None:
            x1, y1, x2, y2 = chosen["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 3)
            draw_label_box(vis, f"Non-Infested (0) | {format_conf(chosen['score'])}", x1, y1)

        summary["status"] = "Non-Infested"
        summary["severity"] = "Non-Infested (0)"
        summary["cut_region"] = "Not needed"
        return vis, summary

    if chosen_class == CLASS_FULL:
        target_dets = full_dets
        chosen_color = (0, 0, 255)
    else:
        target_dets = partial_dets
        chosen_color = (0, 140, 255)

    infested_masks = [d["mask"] for d in target_dets if d["mask"] is not None]
    infested_union = union_masks(infested_masks, vis.shape)

    if infested_union.sum() == 0 and chosen["bbox"] is not None:
        x1, y1, x2, y2 = chosen["bbox"]
        fallback = np.zeros(vis.shape[:2], dtype=np.uint8)
        cv2.rectangle(fallback, (x1, y1), (x2, y2), 1, -1)
        infested_union = fallback

    vis = overlay_mask(vis, infested_union, color=chosen_color, alpha=0.35)

    component_boxes = connected_component_boxes(infested_union, min_area=12)
    for x1, y1, x2, y2, _ in component_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    union_bbox = mask_to_bbox(infested_union)
    if union_bbox is not None:
        ux1, uy1, ux2, uy2 = union_bbox
        cv2.rectangle(vis, (ux1, uy1), (ux2, uy2), (255, 0, 255), 2)

    coverage_pct = (float(infested_union.sum()) / float(vis.shape[0] * vis.shape[1])) * 100.0

    if chosen_class == CLASS_FULL:
        severity = "Severe"
        cut_region = "Not recommended (full infestation)"
    else:
        severity = "Mild" if coverage_pct < 3.5 else "Moderate"
        cut_region = region_name_from_bbox(union_bbox, vis.shape)

    header_text = f"{severity} ({severity_map[severity]}) | {format_conf(chosen['score'])}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 48), (245, 238, 250), -1)
    cv2.putText(vis, header_text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (90, 20, 120), 2, cv2.LINE_AA)

    if union_bbox is not None:
        draw_label_box(vis, header_text, union_bbox[0], max(24, union_bbox[1]))

    summary["status"] = "Infested"
    summary["severity"] = f"{severity} ({severity_map[severity]})"
    summary["cut_region"] = cut_region
    return vis, summary


# =========================================================
# MODEL WRAPPER
# =========================================================
class MMDetSegWrapper:
    def __init__(self, config_path, checkpoint_path, device=DEFAULT_DEVICE):
        self.model = init_detector(str(config_path), str(checkpoint_path), device=device)
        self.classes = [c.strip().lower() for c in self.model.dataset_meta["classes"]]
        print("Loaded classes:", self.classes)

    def infer(self, image):
        result = inference_detector(self.model, image)
        pred = result.pred_instances

        if pred is None or len(pred) == 0:
            return []

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

        detections = []
        for i in range(len(bboxes)):
            cls_name = self.classes[int(labels[i])]
            mask_i = ensure_uint8_mask(masks[i]) if masks is not None and i < len(masks) else None
            detections.append(
                {
                    "label": cls_name,
                    "score": float(scores[i]),
                    "bbox": bboxes[i].astype(int).tolist(),
                    "mask": mask_i,
                }
            )
        return detections


# =========================================================
# MAIN GUI
# =========================================================
class EggplantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eggplant Inspection System")
        self.resize(940, 720)
        self.setMinimumSize(880, 680)

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
        sidebar.setFixedWidth(88)
        sidebar.setObjectName("sidebar")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 16, 10, 16)
        side_layout.setSpacing(14)

        for name in ["HOME", "RESULTS", "ABOUT"]:
            btn = QPushButton(name)
            btn.setMinimumHeight(54)
            side_layout.addWidget(btn)
        side_layout.addStretch()

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(12)

        top_bar = QHBoxLayout()
        title = QLabel("Results")
        title.setObjectName("titleLabel")

        model_text = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_CONFIGS.keys()))
        self.model_combo.setCurrentText(self.current_model_name)
        self.model_combo.currentTextChanged.connect(self.change_model)
        self.model_combo.setFixedWidth(180)

        self.upload_btn = QPushButton("Upload Image")
        self.camera_btn = QPushButton("Use Camera")
        self.analyze_btn = QPushButton("Analyze")

        self.upload_btn.clicked.connect(self.upload_image)
        self.camera_btn.clicked.connect(self.capture_camera)
        self.analyze_btn.clicked.connect(self.run_inspection)

        top_bar.addWidget(title)
        top_bar.addStretch()
        top_bar.addWidget(model_text)
        top_bar.addWidget(self.model_combo)
        top_bar.addWidget(self.upload_btn)
        top_bar.addWidget(self.camera_btn)
        top_bar.addWidget(self.analyze_btn)

        image_card = QFrame()
        image_card.setObjectName("card")
        image_layout = QVBoxLayout(image_card)
        image_layout.setContentsMargins(10, 10, 10, 10)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(DISPLAY_SIZE, DISPLAY_SIZE)
        self.image_label.setStyleSheet("background: white; border-radius: 12px;")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        image_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        result_card = QFrame()
        result_card.setObjectName("card")
        grid = QGridLayout(result_card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        self.status_chip = QLabel("No Result")
        self.status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_chip.setFixedWidth(200)
        self.status_chip.setMinimumHeight(40)

        self.severity_value = QLabel("-")
        self.cut_region_value = QLabel("-")

        for widget in [self.severity_value, self.cut_region_value]:
            widget.setObjectName("valueBox")
            widget.setMinimumHeight(36)

        grid.addWidget(self.status_chip, 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("Severity"), 1, 0)
        grid.addWidget(self.severity_value, 1, 1)
        grid.addWidget(QLabel("Cutting Region"), 2, 0)
        grid.addWidget(self.cut_region_value, 2, 1)

        content_layout.addLayout(top_bar)
        content_layout.addWidget(image_card, alignment=Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(result_card)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(content, 1)

        self.reset_outputs()

    def apply_styles(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{ background: {APP_BG}; }}
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
                padding: 8px 12px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: #8E24AA; }}
            #card {{
                background: {CARD_BG};
                border-radius: 18px;
                padding: 10px;
            }}
            #titleLabel {{
                font-size: 24px;
                font-weight: 700;
                color: #222222;
            }}
            QLabel {{
                font-size: 15px;
                color: #222222;
            }}
            QComboBox {{
                background: white;
                color: #111111;
                border-radius: 10px;
                padding: 6px 8px;
                font-size: 13px;
                min-width: 180px;
            }}
            QComboBox QAbstractItemView {{
                background: white;
                color: #111111;
                selection-background-color: #E9D9F1;
                selection-color: #111111;
            }}
            #valueBox {{
                background: white;
                border-radius: 10px;
                padding: 8px 10px;
                font-size: 14px;
            }}
            QMessageBox {{
                background: #ffffff;
            }}
            QMessageBox QLabel {{
                color: #000000;
                font-size: 13px;
                min-width: 520px;
            }}
            QMessageBox QPushButton {{
                min-width: 80px;
            }}
            """
        )

    def change_model(self, name):
        self.current_model_name = name

    def show_image(self, img):
        pix = bgr_to_pixmap(img)
        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_display_image is not None:
            self.show_image(self.current_display_image)

    def reset_outputs(self):
        self.status_chip.setText("No Result")
        self.status_chip.setStyleSheet(
            "background:#9E9E9E; color:white; border-radius:14px; font-size:17px; font-weight:700; padding:8px 14px;"
        )
        self.severity_value.setText("-")
        self.cut_region_value.setText("-")

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "Load Error", "Failed to read the selected image.")
            return

        self.original_image = fit_image_to_canvas(img)
        self.current_display_image = self.original_image.copy()
        self.show_image(self.current_display_image)
        self.reset_outputs()

    def capture_camera(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            QMessageBox.warning(self, "Camera Error", "Could not open the camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            QMessageBox.warning(self, "Camera Error", "Failed to capture image from camera.")
            return

        self.original_image = fit_image_to_canvas(frame)
        self.current_display_image = self.original_image.copy()
        self.show_image(self.current_display_image)
        self.reset_outputs()

    def get_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        cfg = MODEL_CONFIGS[model_name]

        if not cfg["config"].exists():
            raise FileNotFoundError(
                f"Config file not found.\nPlease check this path:\n{cfg['config']}"
            )

        if not cfg["checkpoint"].exists():
            raise FileNotFoundError(
                f"Checkpoint file not found.\nPlease check this path:\n{cfg['checkpoint']}"
            )

        model = MMDetSegWrapper(cfg["config"], cfg["checkpoint"], DEFAULT_DEVICE)

        found = set(model.classes)
        needed = {CLASS_FULL, CLASS_NON_INFESTED, CLASS_PARTIAL}
        if not needed.issubset(found):
            raise ValueError(
                f"Expected classes: {sorted(needed)}\nFound classes: {sorted(found)}"
            )

        self.loaded_models[model_name] = model
        return model

    def run_inspection(self):
        if self.original_image is None:
            QMessageBox.information(self, "No Image", "Please upload or capture an image first.")
            return

        try:
            model = self.get_model(self.current_model_name)
            detections = model.infer(self.original_image.copy())

            print("Detections:")
            for d in detections:
                print(d["label"], d["score"])

            vis, summary = summarize_detection(self.original_image.copy(), detections)
            self.current_display_image = vis
            self.show_image(vis)
            self.update_result_ui(summary)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Missing File", str(e))
        except ValueError as e:
            QMessageBox.critical(self, "Class Mismatch", str(e))
        except RuntimeError as e:
            QMessageBox.critical(self, "Model Runtime Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))

    def update_result_ui(self, summary):
        if summary["status"] == "Infested":
            self.status_chip.setText("Infested")
            self.status_chip.setStyleSheet(
                "background:#FF4B4B; color:white; border-radius:14px; font-size:17px; font-weight:700; padding:8px 14px;"
            )
        else:
            self.status_chip.setText("Non-Infested")
            self.status_chip.setStyleSheet(
                "background:#22AA55; color:white; border-radius:14px; font-size:17px; font-weight:700; padding:8px 14px;"
            )

        self.severity_value.setText(summary["severity"])
        self.cut_region_value.setText(summary["cut_region"])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EggplantGUI()
    window.show()
    sys.exit(app.exec())