# 🎥 Office Monitoring System with AI YOLOv8


<img width="1916" height="1079" alt="Screenshot 2026-02-10 163718" src="https://github.com/user-attachments/assets/26de4823-e6ec-49ca-8291-39703dcf52c2" />

An intelligent office monitoring system that uses YOLOv8 deep learning model for real-time person detection, tracking, and area occupancy monitoring across multiple office cabin zones. Features both a command-line interface and an interactive Streamlit web dashboard.

## 🌟 Overview


This project implements an automated office monitoring solution that tracks employee presence and movement across different office cabin areas. It leverages the power of YOLOv8 (You Only Look Once) object detection model to identify and track individuals in real-time, calculating the time spent in each designated zone.

**Perfect for:**
- Office space utilization analysis
- Workspace occupancy monitoring
- Security and surveillance applications
- Time tracking in designated areas
- Social distancing compliance monitoring

## ✨ Features

### Core Functionality
- 🤖 **Real-time Person Detection**: Powered by YOLOv8 for accurate and fast detection
- 🎯 **Multi-Area Tracking**: Monitor up to 6 different office cabin zones simultaneously
- ⏱️ **Time Tracking**: Automatic calculation of time spent in each area
- 🔍 **Object Tracking**: Persistent ID assignment for continuous tracking across frames
- 📊 **Visual Analytics**: Real-time statistics overlay on video output

### User Interface
- 🌐 **Web Dashboard**: Interactive Streamlit interface for easy operation
- 📤 **Video Upload**: Support for custom video uploads (MP4, AVI, MOV)
- 📥 **Export Results**: Download processed videos with annotations
- 📈 **Live Statistics**: Real-time cabin occupancy and time metrics
- 🎨 **Color-Coded Zones**: Visual feedback (Green = Occupied, Red = Vacant)

### Technical Features
- ⚡ **GPU Acceleration**: CUDA support for faster processing
- 🔄 **Flexible Input**: Works with live camera feed or pre-recorded videos
- 📐 **Customizable Areas**: Easy polygon definition for monitoring zones
- 💾 **Output Saving**: Automatic saving of processed videos

## 🎬 Demo

### Before Processing
Raw office video feed with multiple people moving around.

### After Processing
- Bounding boxes around detected persons with unique IDs
- Color-coded cabin areas (Green/Red)
- Real-time statistics overlay showing time spent in each cabin
- Person tracking across frames

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Step 1: Clone the Repository
```bash
git clone https://github.com/SHAHFAISAL80/Office-Monitoring-with-ai-yolov8.git
cd Office-Monitoring-with-ai-yolov8
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "from ultralytics import YOLO; import streamlit; print('Installation successful!')"
```

## 💻 Usage

### Method 1: Streamlit Web Interface (Recommended)

1. **Launch the application**:
```bash
streamlit run app.py
```

2. **Access the dashboard**:
   - Open your browser and navigate to `http://localhost:8501`
   - The app will open automatically

3. **Process a video**:
   - Choose "Use Sample Video" or "Upload Custom Video"
   - Click "Start Processing"
   - Monitor real-time progress
   - View results and download processed video

### Method 2: Command Line Interface

1. **Run the main script**:
```bash
python main.py
```

2. **Output**:
   - Processed video saved to `output_video/work_desk_output.mp4`
   - Console output shows detection logs

### Method 3: Custom Video Processing

```python
from main import main

# Process your own video
main(source_video="path/to/your/video.mp4")
```

## 📁 Project Structure

```
Office-Monitoring-with-ai-yolov8/
│
├── app.py                      # Streamlit web application
├── main.py                     # Command-line processing script
├── utilis.py                   # Helper functions (detection, drawing)
├── requirements.txt            # Python dependencies
│
├── input_video/
│   └── work-desk.mp4          # Sample input video
│
├── output_video/              # Processed videos (auto-generated)
│   └── work_desk_output.mp4   # Sample output
│
├── README.md                  # This file
├── README_STREAMLIT.md        # Detailed Streamlit guide
└── QUICKSTART.md              # Quick installation guide
```

## 🔧 How It Works

### 1. Video Input
The system accepts video input from files or live camera feed.

### 2. Object Detection
YOLOv8 processes each frame to detect persons with high accuracy.

### 3. Tracking
Each detected person receives a unique ID that persists across frames using object tracking algorithms.

### 4. Area Monitoring
Predefined polygon zones represent office cabins. The system checks if person centers fall within these zones.

### 5. Time Calculation
For each person in a cabin, the system accumulates time spent using frame duration.

### 6. Visualization
Results are drawn on frames with:
- Bounding boxes and IDs around persons
- Color-coded cabin polygons (Green/Red)
- Statistics overlay showing time per cabin

### 7. Output Generation
Processed frames are compiled into an output video with all annotations.

## ⚙️ Configuration



**Tips for defining areas:**
- Use pixel coordinates from your video frame
- Points should form a closed polygon
- Order matters (clockwise or counter-clockwise)
- Use tools like OpenCV to get coordinates interactively

## 🛠️ Technologies Used

- **[YOLOv8](https://github.com/ultralytics/ultralytics)**: State-of-the-art object detection model
- **[Streamlit](https://streamlit.io/)**: Web framework for the interactive dashboard
- **[OpenCV](https://opencv.org/)**: Computer vision library for video processing
- **[PyTorch](https://pytorch.org/)**: Deep learning framework (backend for YOLOv8)
- **[NumPy](https://numpy.org/)**: Numerical computing library
- **Python 3.8+**: Programming language

## ⚡ Performance

### Processing Speed

| Hardware | FPS | Speed |
|----------|-----|-------|
| CPU (Intel i7) | ~5-10 | Slow |
| GPU (NVIDIA GTX 1660) | ~30-40 | Good |
| GPU (NVIDIA RTX 3080) | ~60-80 | Excellent |

## 🐛 Troubleshooting

### Issue: YOLOv8 model not downloading

**Solution**:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Issue: Streamlit port already in use

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

### Issue: CUDA out of memory

**Solution**:
- Reduce video resolution
- Use smaller YOLOv8 model (yolov8n.pt is smallest)
- Process fewer frames

### Issue: Video codec error

**Solution**:
```bash
pip install opencv-python-headless
```

### Issue: Slow processing on CPU

**Solution**:
- Install GPU version of PyTorch
- Reduce video resolution
- Use yolov8n.pt (nano model)
- Process fewer frames

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contributions
- Add support for multiple camera angles
- Implement alert system for unauthorized areas
- Add heatmap visualization
- Export statistics to CSV/Excel
- Add face recognition integration
- Implement real-time email notifications
- Add database storage for historical data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Shah Faisal**

- GitHub: [@SHAHFAISAL80](https://github.com/SHAHFAISAL80)
- Project Link: [https://github.com/SHAHFAISAL80/Office-Monitoring-with-ai-yolov8](https://github.com/SHAHFAISAL80/Office-Monitoring-with-ai-yolov8)

## 🙏 Acknowledgments

- **Ultralytics**: For the amazing YOLOv8 implementation
- **Streamlit Team**: For the powerful web framework
- **OpenCV Community**: For comprehensive computer vision tools
- **PyTorch Team**: For the robust deep learning framework

## 📊 Stats

![GitHub Stars](https://img.shields.io/github/stars/SHAHFAISAL80/Office-Monitoring-with-ai-yolov8?style=social)
![GitHub Forks](https://img.shields.io/github/forks/SHAHFAISAL80/Office-Monitoring-with-ai-yolov8?style=social)
![GitHub Issues](https://img.shields.io/github/issues/SHAHFAISAL80/Office-Monitoring-with-ai-yolov8)

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ by Shah Faisal

</div>
