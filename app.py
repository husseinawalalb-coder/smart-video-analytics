
import streamlit as st
import cv2
import tempfile
import os
import pandas as pd

from ultralytics import YOLO
from collections import Counter

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Video Analytics",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Smart Video Analytics")
st.write(
    "Upload a video to detect, track, and count objects "
    "using YOLO11 and Computer Vision."
)

st.divider()


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# ==========================================
# 3. SIDEBAR SETTINGS
# ==========================================

st.sidebar.header("⚙️ Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

st.sidebar.info(
    "Lower confidence detects more objects, "
    "but may produce more false detections."
)


# ==========================================
# 4. VIDEO UPLOADER
# ==========================================

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)


# ==========================================
# 5. VIDEO PROCESSING FUNCTION
# ==========================================

def process_video(input_path, output_path, confidence):

    video = cv2.VideoCapture(input_path)

    if not video.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25

    if width <= 0 or height <= 0:
        video.release()
        raise ValueError("Invalid video dimensions.")

    # MP4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output_video = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not output_video.isOpened():
        video.release()
        raise ValueError("Could not create the output video.")

    # Track unique IDs
    counted_ids = set()

    # Store the class of each tracked object
    tracked_classes = {}

    frame_count = 0

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Reset tracking state for this video
    model.predictor = None

    try:
        while True:

            success, frame = video.read()

            if not success:
                break

            frame_count += 1

            # Detect and track objects
            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # Draw bounding boxes and tracking IDs
            annotated_frame = result.plot()

            boxes = result.boxes

            # Extract IDs and object classes
            if boxes.id is not None:

                tracking_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()

                for track_id, class_id in zip(
                    tracking_ids,
                    class_ids
                ):

                    counted_ids.add(track_id)

                    class_name = model.names[class_id]

                    if track_id not in tracked_classes:
                        tracked_classes[track_id] = class_name

            # Total unique tracked objects
            total_objects = len(counted_ids)

            # Draw total on video
            cv2.putText(
                annotated_frame,
                f"Total Objects: {total_objects}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Save processed frame
            output_video.write(annotated_frame)

            # Update progress
            if total_frames > 0:
                progress = min(
                    frame_count / total_frames,
                    1.0
                )
                progress_bar.progress(progress)

            status_text.text(
                f"Processing frame {frame_count}"
            )

    finally:
        video.release()
        output_video.release()

    progress_bar.progress(1.0)
    status_text.success("Video processing completed!")

    # Count objects by class
    class_counts = Counter(
        tracked_classes.values()
    )

    return total_objects, class_counts, frame_count


# ==========================================
# 6. RUN ANALYSIS
# ==========================================

if uploaded_video is not None:

    st.subheader("📹 Original Video")

    st.video(uploaded_video)

    if st.button(
        "🚀 Start Analysis",
        type="primary"
    ):

        with st.spinner("Analysing video..."):

            input_path = None
            output_path = None

            try:
                # Save uploaded video temporarily
                suffix = os.path.splitext(
                    uploaded_video.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_input:

                    temp_input.write(
                        uploaded_video.getvalue()
                    )

                    input_path = temp_input.name

                # Create output file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_output:

                    output_path = temp_output.name

                # Process video
                total_objects, class_counts, frame_count = (
                    process_video(
                        input_path,
                        output_path,
                        confidence
                    )
                )

                # Read processed video
                with open(output_path, "rb") as f:
                    processed_video = f.read()

                # Store results
                st.session_state["analysis"] = {
                    "video": processed_video,
                    "total": total_objects,
                    "counts": dict(class_counts),
                    "frames": frame_count
                }

            except Exception as e:
                st.error(f"Processing failed: {e}")

            finally:
                # Delete temporary files
                for path in [input_path, output_path]:
                    if path and os.path.exists(path):
                        os.remove(path)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

if "analysis" in st.session_state:

    analysis = st.session_state["analysis"]

    st.divider()

    st.header("📊 Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Unique Objects",
            analysis["total"]
        )

    with col2:
        st.metric(
            "Frames Processed",
            analysis["frames"]
        )

    st.subheader("🎬 Processed Video")

    st.video(analysis["video"])

    st.subheader("📈 Objects by Class")

    if analysis["counts"]:

        df = pd.DataFrame(
            [
                {
                    "Object Class": name,
                    "Count": count
                }
                for name, count in analysis["counts"].items()
            ]
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            df.set_index("Object Class")["Count"]
        )

    else:
        st.warning(
            "No tracked objects were found in the video."
        )

    st.download_button(
        label="⬇️ Download Processed Video",
        data=analysis["video"],
        file_name="analysed_video.mp4",
        mime="video/mp4"
    )


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Smart Video Analytics | YOLO11 + OpenCV + Streamlit"
)
