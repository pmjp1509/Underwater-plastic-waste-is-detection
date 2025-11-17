# YOLOv8 Plastic Detection System

A deep learning-based object detection system that identifies and classifies three types of plastic waste: **Masks**, **Bags**, and **Bottles**. Built with YOLOv8 (Nano, Medium, Large variants) and deployed with a user-friendly Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Features

✅ **Three Model Variants**
   - Nano (lightweight, fast inference)
   - Medium (balanced accuracy & speed)
   - Large (highest accuracy, slower inference)

✅ **Real-Time Detection**
   - Fast inference on CPU and GPU
   - Adjustable confidence threshold
   - Configurable image resolution (320px, 480px, 640px, 800px)

✅ **User-Friendly Interface**
   - Web-based Streamlit dashboard
   - Simple image upload and detection
   - Visual bounding boxes on detected plastics
   - Detection summary with counts

✅ **Production-Ready**
   - Pre-trained models on custom plastic dataset
   - Balanced and augmented training data
   - Optimized for real-world scenarios

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Model Details](#model-details)
6. [Dataset](#dataset)
7. [Configuration](#configuration)
8. [Performance](#performance)
9. [Troubleshooting](#troubleshooting)
10. [Applications & Future Scope](#applications--future-scope)
11. [Contributing](#contributing)
12. [License](#license)

---

## 🛠️ Installation

### Prerequisites
- **Python** 3.8 or higher
- **pip** (Python package manager)
- **Windows/Linux/macOS** (tested on Windows)

### Step 1: Clone or Download the Project
```bash
cd AI_PROJECT
```

### Step 2: Create Virtual Environment (Recommended)
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `ultralytics` - YOLOv8 framework
- `streamlit` - Web interface
- `opencv-python-headless` - Image processing
- `pillow` - Image manipulation
- `numpy` - Numerical computations
- `pandas` - Data handling
- `matplotlib` - Visualization
- `pyyaml` - Configuration files
- `tqdm` - Progress bars

### Step 4: Verify Installation
```bash
python -c "from ultralytics import YOLO; print('✅ YOLOv8 installed successfully!')"
```

---

## 🚀 Quick Start

### Run the Web Application

```powershell
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

### Using the Interface

1. **Select Model**: Choose from Nano, Medium, or Large in the sidebar
2. **Upload Image**: Click "Upload an image" and select a JPG, JPEG, or PNG file
3. **Adjust Parameters**:
   - Confidence threshold (0.0 - 1.0, default: 0.25)
   - Image size (320px, 480px, 640px, 800px)
4. **Run Detection**: Click "Run detection" button
5. **View Results**:
   - Annotated image with bounding boxes
   - Detection summary showing count of each plastic type

### Example Output
```
Result
[Annotated image with bounding boxes]

Detections Found
🔍 Masks: 5 detected
🔍 Bags: 3 detected
🔍 Bottles: 7 detected
```

---

## 📁 Project Structure

```
AI_PROJECT/
├── app.py                           # Streamlit web application
├── Project.ipynb                    # Jupyter notebook (Colab version)
├── requirements.txt                 # Python dependencies
├── nano_best.pt                     # YOLOv8 Nano trained model
├── medium_best.pt                   # YOLOv8 Medium trained model
├── large_best.pt                    # YOLOv8 Large trained model
├── .venv/                           # Virtual environment (created during setup)
└── README.md                        # Environment setup guide
```

---

## 💻 Usage

### Command-Line Inference (Advanced)

```python
from ultralytics import YOLO

# Load model
model = YOLO('nano_best.pt')  # or 'medium_best.pt' or 'large_best.pt'

# Run inference
results = model.predict(source='image.jpg', conf=0.25, imgsz=640)

# Access detections
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = result.names[class_id]
        print(f"{class_name}: {confidence:.2f}")
```

### Batch Processing

```python
from ultralytics import YOLO

model = YOLO('medium_best.pt')

# Process entire folder
results = model.predict(
    source='path/to/images/',
    conf=0.25,
    imgsz=640,
    save=True  # Save annotated images
)
```

### Video Processing

```python
from ultralytics import YOLO

model = YOLO('medium_best.pt')

# Process video
results = model.predict(
    source='video.mp4',
    conf=0.25,
    imgsz=640,
    save=True  # Save output video
)
```

---

## 🧠 Model Details

### YOLOv8 Variants Comparison

| Aspect | Nano | Medium | Large |
|--------|------|--------|-------|
| **Model Size** | ~7 MB | ~50 MB | ~200 MB |
| **Inference Time** | ~15ms | ~40ms | ~100ms |
| **Memory Usage** | ~100 MB | ~500 MB | ~2 GB |
| **CPU Inference** | ✅ Fast | ⚠️ Moderate | ❌ Slow |
| **GPU Inference** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Accuracy** | Good | Better | Best |
| **Best For** | Edge devices, Real-time | Balanced use | High accuracy |

### Training Details

**Dataset:**
- Source: plastics_filtered dataset
- Train/Val/Test split: standard YOLO format
- Classes: 3 (Masks, Bags, Bottles)

**Training Configuration:**
- Framework: YOLOv8
- Epochs: 25
- Image Size: 640×640
- Batch Sizes:
  - Nano: 32
  - Medium: 16
  - Large: 8
- Data Augmentation: Flip, Rotate, Brightness adjustment
- Balanced Dataset: 650 images per class after augmentation

**Training Paths:**
```
/content/AI_Project/
├── YOLOv8_Nano/yolov8-nano_plastics_balanced/weights/best.pt
├── YOLOv8_Medium/yolov8-medium_plastics_balanced/weights/best.pt
└── YOLOv8_Large/yolov8-large_plastics_balanced/weights/best.pt
```

---

## 📊 Dataset

### Classes

| Class | Description | Examples |
|-------|-------------|----------|
| **Masks** | Face masks, protective masks | N95 masks, cloth masks, surgical masks |
| **Bags** | Plastic bags | Shopping bags, garbage bags, carrier bags |
| **Bottles** | Plastic bottles | Water bottles, beverage containers, soda bottles |

### Dataset Statistics

- **Total Images**: ~1,950 (650 per class after balancing)
- **Original Classes**: Imbalanced
- **After Augmentation**: Balanced (650 images each)
- **Augmentation Methods**:
  - Horizontal flip
  - Random rotation (90°, 180°, 270°)
  - Brightness adjustment (0.7x to 1.3x)
- **Format**: YOLO format (images + labels)

### Data Verification

The training pipeline includes automatic verification:
```
✅ Images without labels: 0
✅ Labels without images: 0
✅ All pairs matched correctly
```

---

## ⚙️ Configuration

### Streamlit Configuration

Modify inference parameters in the web interface:

```python
# Confidence Threshold
- Range: 0.0 - 1.0
- Default: 0.25
- Higher = fewer detections, higher precision
- Lower = more detections, higher recall

# Image Size
- Options: 320, 480, 640, 800 (pixels)
- Default: 640
- Larger = better accuracy, slower inference
- Smaller = faster inference, lower accuracy
```

### Model Selection

Three models available in sidebar dropdown:
- **Nano (nano_best.pt)** - Fastest, good accuracy
- **Medium (medium_best.pt)** - Balanced performance
- **Large (large_best.pt)** - Highest accuracy

---

## 📈 Performance

### Inference Speed (on CPU)

| Model | Image Size | Time | FPS |
|-------|-----------|------|-----|
| Nano | 640×640 | ~200ms | 5 |
| Medium | 640×640 | ~400ms | 2.5 |
| Large | 640×640 | ~800ms | 1.2 |

*Note: Actual performance depends on your hardware. GPU inference is significantly faster.*

### Accuracy Metrics

- **Nano**: Good performance on small-medium objects
- **Medium**: Better small object detection, balanced
- **Large**: Highest accuracy on all scales

### Detection Examples

**Input**: Image with mixed plastic waste  
**Output**:
```
Masks: 106 detected
Bags: 74 detected
Bottles: 120 detected
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Verify virtual environment is activated
.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Streamlit app won't start

**Solution:**
```bash
# Try with verbose output
streamlit run app.py --logger.level=debug

# Clear Streamlit cache
streamlit cache clear
```

### Issue: Slow inference on CPU

**Solution:**
- Use **Nano model** for faster inference
- Reduce **image size** (try 480px or 320px)
- Increase **confidence threshold** to filter low-confidence detections
- (Optional) Install GPU support (CUDA for Nvidia, ROCm for AMD)

### Issue: GPU not detected

**Solution:**
```python
# Check if GPU is available
from ultralytics import YOLO

model = YOLO('nano_best.pt')
model.to('cuda')  # Force CUDA
```

### Issue: Out of memory errors

**Solution:**
- Use **Nano model**
- Reduce image size
- Process fewer images at once
- Reduce batch size (in training scripts)

### Issue: Inaccurate detections

**Solution:**
- Try **Medium or Large model** for better accuracy
- Ensure good lighting in images
- Increase **image resolution** (use 640px or 800px)
- Reduce **confidence threshold** if missing objects

---

## 🎓 Applications & Future Scope

### Current Applications
- Waste management and recycling automation
- Environmental monitoring (beaches, oceans, landfills)
- Smart city infrastructure
- Manufacturing quality control
- Educational purposes

### Future Enhancements
- **Phase 1**: Expand to 10+ plastic types, add material classification
- **Phase 2**: REST API, cloud deployment, mobile app
- **Phase 3**: Video processing, 3D detection, multi-modal learning
- **Phase 4**: Autonomous robots, IoT integration, satellite imagery

📖 **See `APPLICATIONS_AND_SCOPE.md` for detailed roadmap**

---

## 🤝 Contributing

### How to Contribute

1. **Report Issues**: Found a bug? Open an issue on GitHub
2. **Improve Data**: Contribute more training images for better accuracy
3. **Optimize Code**: Submit pull requests with improvements
4. **Documentation**: Help improve documentation and README
5. **Feature Requests**: Suggest new features or capabilities

### Development Setup

```bash
# Clone repo
git clone <repository-url>
cd AI_PROJECT

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Development tools

# Run tests
pytest tests/
```

---

## 📜 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## 📞 Support & Contact

### Getting Help

1. **Check Troubleshooting Section** above
2. **Read APPLICATIONS_AND_SCOPE.md** for use cases
3. **View example inference** in the Streamlit app
4. **Consult YOLOv8 documentation**: https://docs.ultralytics.com

### Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com
- **Streamlit Documentation**: https://docs.streamlit.io
- **YOLO GitHub**: https://github.com/ultralytics/ultralytics

---

## 🎉 Quick Reference

### Start the app
```bash
streamlit run app.py
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run inference from Python
```python
from ultralytics import YOLO
model = YOLO('nano_best.pt')
results = model.predict('image.jpg')
```

### View performance metrics
- Check terminal output during inference
- Models show: **Speed** (preprocess, inference, postprocess), **Total FPS**

---

## 📚 Project Info

**Version**: 1.0  
**Created**: November 2024  
**Framework**: YOLOv8 (Ultralytics)  
**Interface**: Streamlit  
**Python Version**: 3.8+  
**Status**: ✅ Production Ready

---

## 🚀 Next Steps

1. ✅ Run the Streamlit app: `streamlit run app.py`
2. ✅ Upload a plastic image and test detection
3. ✅ Try different models (Nano, Medium, Large)
4. ✅ Experiment with confidence thresholds
5. ✅ Explore APPLICATIONS_AND_SCOPE.md for future ideas

---

**Happy Detecting! 🎯🗑️♻️**

For questions or suggestions, feel free to contribute or create an issue.
