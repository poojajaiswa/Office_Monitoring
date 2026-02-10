import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection, draw_working_areas
import tempfile
import os
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Office Monitoring System",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def setup_device():
    """Check if CUDA is available and set the device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


@st.cache_resource
def load_yolo_model(device):
    """Load the YOLO model and configure it."""
    model = YOLO("yolov8n.pt")
    model.to(device)
    model.nms = 0.5
    return model


def initialize_variables(num_areas):
    """Initialize time tracking variables."""
    time_in_area = {index: 0 for index in range(num_areas)}
    entry_time = {}
    return time_in_area, entry_time


def calculate_center(box):
    """Calculate the center of a bounding box."""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)


def track_time(id, index, frame_cnt, entry_time, time_in_area, frame_duration):
    """Track time spent by each object in each working area."""
    if id not in entry_time:
        entry_time[id] = (frame_cnt, index)
    else:
        start_frame, area_index = entry_time[id]
        if area_index != index:
            time_in_area[area_index] += frame_duration
            entry_time[id] = (frame_cnt, index)
        else:
            time_in_area[area_index] += frame_duration


def draw_polygons(frame, working_area, polygon_detections):
    """Draw working areas with specific color coding based on detections."""
    for index, pos in enumerate(working_area):
        color = (0, 255, 0) if polygon_detections[index] else (0, 0, 255)
        draw_working_areas(frame=frame, area=pos, index=index, color=color)


def display_time_overlay(frame, time_in_area):
    """Overlay time spent in each area on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 250), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    for index, time_spent in time_in_area.items():
        cv2.putText(frame, f"Cabin {index + 1}: {round(time_spent)}s", (15, 30 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def process_frame(model, frame, working_area, time_in_area, entry_time, frame_cnt, frame_duration):
    """Process a single frame to detect objects and track time spent in each area."""
    boxes, classes, names, confidences, ids = YOLO_Detection(model, frame, conf=0.05, mode="track")
    polygon_detections = [False] * len(working_area)

    for box, cls, id in zip(boxes, classes, ids):
        center_point = calculate_center(box)
        label_detection(frame=frame, text=f"{names[int(cls)]}, {int(id)}", tbox_color=(255, 144, 30), 
                       left=box[0], top=box[1], bottom=box[2], right=box[3])

        for index, pos in enumerate(working_area):
            if cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False) >= 0:
                polygon_detections[index] = True
                track_time(id, index, frame_cnt, entry_time, time_in_area, frame_duration)

    draw_polygons(frame, working_area, polygon_detections)
    display_time_overlay(frame, time_in_area)
    
    return frame


def process_video(video_path, model, working_area, progress_bar, status_text):
    """Process the entire video and return output path."""
    time_in_area, entry_time = initialize_variables(len(working_area))
    frame_duration = 0.1
    
    cap = cv2.VideoCapture(video_path)
    frame_cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if it doesn't exist
    output_dir = Path("output_video")
    output_dir.mkdir(exist_ok=True)
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = output_dir / f"processed_{int(time.time())}.mp4"
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_cnt += 1
        processed_frame = process_frame(model, frame, working_area, time_in_area, 
                                       entry_time, frame_cnt, frame_duration)
        out.write(processed_frame)
        
        # Update progress
        progress = frame_cnt / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_cnt}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path, time_in_area


def main():
    st.title("🎥 Office Monitoring System with YOLOv8")
    st.markdown("Track people's presence in different office areas using AI-powered object detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Device information
        device = setup_device()
        device_name = "GPU (CUDA)" if device.type == "cuda" else "CPU"
        st.info(f"**Device:** {device_name}")
        
        st.markdown("---")
        
        # Working area configuration
        st.subheader("📍 Working Areas")
        st.markdown("""
        The system monitors 6 predefined cabin areas:
        - **Cabin 1-6**: Different office zones
        - **Green**: Area occupied
        - **Red**: Area vacant
        """)
        
        # Video source selection
        st.markdown("---")
        st.subheader("📹 Video Source")
        video_source = st.radio(
            "Choose video source:",
            ["Use Sample Video", "Upload Custom Video"],
            index=0
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Video Processing")
        
        # Video input
        if video_source == "Use Sample Video":
            video_path = "input_video/work-desk.mp4"
            if os.path.exists(video_path):
                st.success("✅ Using sample video: work-desk.mp4")
            else:
                st.error("❌ Sample video not found. Please upload a video.")
                video_path = None
        else:
            uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
            if uploaded_file is not None:
                # Save uploaded file to temp directory
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                st.success(f"✅ Uploaded: {uploaded_file.name}")
            else:
                video_path = None
        
        # Process button
        if video_path and st.button("🚀 Start Processing", type="primary", use_container_width=True):
            # Define working areas (office cabins)
            working_area = [
                [(499, 41), (384, 74), (377, 136), (414, 193), (417, 112), (548, 91)],
                [(547, 91), (419, 113), (414, 189), (452, 289), (453, 223), (615, 164)],
                [(158, 84), (294, 85), (299, 157), (151, 137)],
                [(151, 139), (300, 155), (321, 251), (143, 225)],
                [(143, 225), (327, 248), (351, 398), (142, 363)],
                [(618, 166), (457, 225), (454, 289), (522, 396), (557, 331), (698, 262)]
            ]
            
            with st.spinner("Loading YOLO model..."):
                model = load_yolo_model(device)
            
            st.info("🔄 Processing video... This may take a few minutes.")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Process video
                output_path, time_in_area = process_video(
                    video_path, model, working_area, progress_bar, status_text
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                # Display results
                st.success("🎉 Video processed successfully!")
                
                # Show processed video
                st.subheader("📺 Processed Video")
                st.video(str(output_path))
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f,
                        file_name=f"office_monitoring_{int(time.time())}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
    
    with col2:
        st.header("📊 Statistics")
        
        if 'time_in_area' in locals():
            st.subheader("⏱️ Time in Each Area")
            
            # Create statistics display
            for cabin_num, time_spent in time_in_area.items():
                minutes = int(time_spent // 60)
                seconds = int(time_spent % 60)
                
                st.metric(
                    label=f"Cabin {cabin_num + 1}",
                    value=f"{minutes}m {seconds}s",
                    delta=f"{round(time_spent)}s total"
                )
        else:
            st.info("Process a video to see statistics")
        
        st.markdown("---")
        
        # Information
        st.subheader("ℹ️ About")
        st.markdown("""
        This system uses **YOLOv8** for real-time object detection and tracking.
        
        **Features:**
        - Person detection and tracking
        - Area occupancy monitoring
        - Time tracking per cabin
        - Visual annotations
        
        **How it works:**
        1. Upload or use sample video
        2. Click 'Start Processing'
        3. View results and download
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Office Monitoring System | Powered by YOLOv8 & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
