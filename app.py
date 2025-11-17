import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import pandas as pd


# Path to trained models
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS = {
    "Nano (nano_best.pt)": os.path.join(MODELS_DIR, "nano_best.pt"),
    "Medium (medium_best.pt)": os.path.join(MODELS_DIR, "medium_best.pt"),
    "Large (large_best.pt)": os.path.join(MODELS_DIR, "large_best.pt"),
}


@st.cache_resource
def load_model(weights_path: str):
    """Load YOLO model (cached). Supports local path or model name."""
    return YOLO(weights_path)


def annotate_results(results):
    """Return annotated image (numpy array) and class names detected."""
    r = results[0]
    # annotated image (np.ndarray)
    ann = r.plot()  # ultralytics provides a plotted image

    # Extract class names
    class_counts = {}
    if hasattr(r, 'boxes') and r.boxes is not None:
        try:
            # Get class names from the model
            class_names = r.names  # dict or list of class names
            
            # Count detections by class
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = class_names.get(cls_id, f"Class {cls_id}") if isinstance(class_names, dict) else class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        except Exception:
            # fallback: empty
            class_counts = {}

    return ann, class_counts


def main():
    st.set_page_config(page_title="YOLOv8 Demo", layout="centered")
    st.title("YOLOv8 — Plastic Detection")

    st.markdown(
        "Upload an image and run detection using one of your trained models."
    )

    # Put controls in a narrow left column and main content on the right
    col1, col2 = st.columns([1, 3])

    with col1:
        model_choice = st.selectbox("Select Model", list(TRAINED_MODELS.keys()), index=0)
        weights = TRAINED_MODELS[model_choice]
        conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25)
        imgsz = st.selectbox("Image size (px)", [320, 480, 640, 800], index=2)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) 

    if uploaded is not None:
        st.image(uploaded, caption="Input image")

    if st.button("Run detection"):
        if uploaded is None:
            st.warning("Please upload an image first.")
            return

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # Load model (cached)
        try:
            model = load_model(weights)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

        # Run prediction
        try:
            results = model.predict(source=tmp_path, conf=conf, imgsz=imgsz)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

        ann, class_counts = annotate_results(results)

        st.subheader("Result")
        # Ultraytics' plot returns an np.ndarray (RGB). Show it directly.
        st.image(ann)

        if class_counts:
            st.subheader("Detections Found")
            for cls_name, count in class_counts.items():
                st.write(f"🔍 **{cls_name}**: {count} detected")
        else:
            st.info("No detections above the confidence threshold.")

        # Cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()
