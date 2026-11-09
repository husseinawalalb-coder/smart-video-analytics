
import cv2
from ultralytics import YOLO

# Step 1: Load YOLO model
model = YOLO("yolo11n.pt")

# Store unique tracking IDs
counted_ids = set()

# Step 2: Open the video
video = cv2.VideoCapture("videos/test.mp4")

# Step 3: Check whether the video opened
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Step 4: Process frames
while True:
    success, frame = video.read()

    if not success:
        break

    # Step 5: Detect and track objects
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # Step 6: Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Get tracking IDs
    boxes = results[0].boxes

    if boxes.id is not None:
        tracking_ids = boxes.id.int().cpu().tolist()

        for track_id in tracking_ids:
            counted_ids.add(track_id)

        print("Unique objects:", len(counted_ids))

    # Step 7: Display total unique objects
    total_objects = len(counted_ids)

    cv2.putText(
        annotated_frame,
        f"Total Objects: {total_objects}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Step 8: Display the frame
    cv2.imshow("Smart Video Analytics", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 9: Release resources
video.release()
cv2.destroyAllWindows()

print("Video processing completed!")
