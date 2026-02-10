# Office Monitoring System with YOLOv8 and Streamlit

A real-time office monitoring system that uses YOLOv8 for person detection and tracking across multiple office cabin areas. Features an intuitive Streamlit web interface for easy video processing and visualization.

## 🌟 Features

- **Real-time Person Detection**: Uses YOLOv8 for accurate person detection
- **Multi-Area Tracking**: Monitor up to 6 different office cabin areas
- **Time Tracking**: Automatically track time spent in each area
- **Web Interface**: User-friendly Streamlit dashboard
- **Video Processing**: Upload custom videos or use sample footage
- **Visual Annotations**: Color-coded areas (Green = Occupied, Red = Vacant)
- **Export Results**: Download processed videos with annotations

## 📋 Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster processing)

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Run Streamlit App (Recommended)

1. **Start the Streamlit application**:
```bash
streamlit run app.py
```

2. **Open your browser** and navigate to:
```
http://localhost:8501
```

3. **Use the interface**:
   - Choose between sample video or upload your own
   - Click "Start Processing" to begin analysis
   - View real-time progress and statistics
   - Download the processed video

### Option 2: Run Command Line Version

```bash
python main.py
```

This will process the sample video in `input_video/work-desk.mp4` and save the output to `output_video/`.

## 📁 Project Structure

```
Office-Monitoring-with-ai-yolov8-master/
├── app.py                      # Streamlit web application (NEW)
├── main.py                     # Command-line processing script
├── utilis.py                   # Helper functions for detection and visualization
├── requirements.txt            # Python dependencies (NEW)
├── input_video/
│   └── work-desk.mp4          # Sample input video
├── output_video/              # Processed videos (auto-generated)
└── README.md                  # This file
```

## 🎯 How It Works

### Detection & Tracking
- YOLOv8 model detects and tracks people in each frame
- Each person receives a unique ID for continuous tracking
- Bounding boxes and labels show detected individuals

### Area Monitoring
- 6 predefined polygon areas represent office cabins
- System tracks when people enter/exit each area
- Color coding: 🟢 Green (occupied) | 🔴 Red (vacant)

### Time Tracking
- Accumulates time each person spends in each cabin
- Real-time statistics displayed on video overlay
- Summary available in Streamlit dashboard

## 🔧 Configuration

### Modify Working Areas
Edit the `working_area` list in `app.py` or `main.py` to customize cabin boundaries:

```python
working_area = [
    [(x1, y1), (x2, y2), ...],  # Cabin 1
    [(x1, y1), (x2, y2), ...],  # Cabin 2
    # ... add more areas
]
```

### Adjust Detection Parameters
In `utilis.py`, modify:
- `conf`: Confidence threshold (default: 0.05)
- `iou`: Intersection over Union threshold (default: 0.1)

## 📊 Streamlit Interface Guide

### Sidebar
- **Device Status**: Shows if using GPU or CPU
- **Working Areas Info**: Configuration details
- **Video Source**: Choose sample or upload custom video

### Main Panel
- **Video Processing**: Upload and process videos
- **Live Progress**: Real-time processing status
- **Results Display**: View processed video inline
- **Download**: Get processed video file

### Statistics Panel
- **Time Metrics**: Time spent in each cabin
- **Occupancy Data**: Real-time area statistics
- **About Section**: System information

## 🎥 Video Format Support

Supported formats:
- MP4
- AVI
- MOV

Recommended specifications:
- Resolution: 720p or higher
- Frame rate: 30 fps
- Good lighting conditions for better detection

## ⚡ Performance Tips

1. **Use GPU**: Install CUDA-enabled PyTorch for 10x faster processing
2. **Video Resolution**: Lower resolution = faster processing
3. **Reduce FPS**: Process every nth frame for quicker results
4. **Confidence Threshold**: Adjust for accuracy vs speed trade-off

## 🐛 Troubleshooting

### Model Download Issues
If YOLOv8 model doesn't download automatically:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Streamlit Port Already in Use
Change the port:
```bash
streamlit run app.py --server.port 8502
```

### Video Codec Issues
Install additional codecs:
```bash
pip install opencv-python-headless
```

### CUDA/GPU Not Detected
Ensure CUDA toolkit is installed and PyTorch has CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📧 Contact

For questions or support, please open an issue on the GitHub repository.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the amazing object detection model
- **Streamlit**: For the powerful web framework
- **OpenCV**: For video processing capabilities

---

**Note**: The YOLOv8 model will be automatically downloaded on first run (~6MB).
