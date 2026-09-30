import cv2
import json
import time
from ultralytics import YOLO

from navigation import (
    Detection,
    estimate_proximity,
    get_navigation_message,
)


# ============================================================
# VisionGuide
# Camera → YOLO → Scene Understanding → scene.json
# ============================================================
import os
MODEL_NAME = "yolo11n.pt"
CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.45
REPORT_INTERVAL = 1.0

SCENE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scene.json"
)


# Objects that matter for navigation
IMPORTANT_OBJECTS = {
    "person",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "truck",
    "dog",
    "chair",
    "bench",
    "backpack",
    "cell phone",
    "bottle",
}


# ============================================================
# Load model
# ============================================================

print("🧠 Loading VisionGuide AI model...")

model = YOLO(MODEL_NAME)

print("✅ YOLO model loaded")


# ============================================================
# Start camera
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera")

print("📷 Camera started")
print("🧭 VisionGuide is ACTIVE")
print("📄 Scene data →", SCENE_FILE)
print("❌ Press Q to quit")


last_report = 0


# ============================================================
# Main loop
# ============================================================

while True:

    # --------------------------------------------------------
    # Capture frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read camera frame")
        break

    frame_height, frame_width = frame.shape[:2]

    frame_area = frame_height * frame_width


    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    results = model(
        frame,
        verbose=False
    )

    result = results[0]

    detections = []


    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])

            confidence = float(box.conf[0])

            label = model.names[cls_id]


            # Ignore weak detections
            if confidence < CONFIDENCE_THRESHOLD:
                continue


            # Ignore irrelevant objects
            if label not in IMPORTANT_OBJECTS:
                continue


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1

            box_area = box_width * box_height


            # ------------------------------------------------
            # Horizontal position
            # ------------------------------------------------

            center_x = (x1 + x2) / 2


            if center_x < frame_width * 0.33:

                position = "left"

            elif center_x > frame_width * 0.66:

                position = "right"

            else:

                position = "center"


            # ------------------------------------------------
            # Approximate proximity
            # ------------------------------------------------

            proximity = estimate_proximity(
                box_area,
                frame_area
            )


            # ------------------------------------------------
            # Create structured detection
            # ------------------------------------------------

            detections.append(
                Detection(
                    object=label,
                    confidence=confidence,
                    position=position,
                    proximity=proximity,
                )
            )


    # ========================================================
    # Generate scene information
    # ========================================================

    now = time.time()


    if now - last_report >= REPORT_INTERVAL:

        # ----------------------------------------------------
        # Navigation message
        # ----------------------------------------------------

        guidance = get_navigation_message(
            detections
        )


        # ----------------------------------------------------
        # Convert detections to JSON
        # ----------------------------------------------------

        detection_data = []

        for detection in detections:

            detection_data.append(
                {
                    "object": detection.object,
                    "confidence": round(
                        detection.confidence,
                        2
                    ),
                    "position": detection.position,
                    "proximity": detection.proximity,
                }
            )


        # ----------------------------------------------------
        # Complete scene
        # ----------------------------------------------------

        scene_data = {
            "timestamp": now,
            "guidance": guidance,
            "detections": detection_data,
            "camera": {
                "width": frame_width,
                "height": frame_height,
            },
        }


        # ----------------------------------------------------
        # Save scene for LiveKit
        # ----------------------------------------------------

        try:

            with open(
                SCENE_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    scene_data,
                    file,
                    indent=2
                )

        except Exception as e:

            print(
                f"⚠️ Could not write scene.json: {e}"
            )


        # ====================================================
        # Terminal output
        # ====================================================

        print()
        print("=" * 60)

        print(
            f"🧭 GUIDANCE: {guidance}"
        )

        print("-" * 60)


        if detections:

            for detection in detections:

                print(
                    f"👁️ "
                    f"{detection.object:<12} | "
                    f"{detection.position:<6} | "
                    f"{detection.proximity:<10} | "
                    f"{detection.confidence:.2f}"
                )

        else:

            print(
                "✅ No important objects detected"
            )


        print("=" * 60)


        last_report = now


    # ========================================================
    # Draw YOLO detections
    # ========================================================

    annotated_frame = result.plot()


    # --------------------------------------------------------
    # VisionGuide status
    # --------------------------------------------------------

    cv2.putText(
        annotated_frame,
        "VISIONGUIDE ACTIVE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )


    # --------------------------------------------------------
    # Current guidance on screen
    # --------------------------------------------------------

    if detections:

        current_guidance = get_navigation_message(
            detections
        )

    else:

        current_guidance = "Path appears clear"


    # Keep text reasonably short
    display_text = current_guidance[:70]


    cv2.putText(
        annotated_frame,
        display_text,
        (20, frame_height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


    # ========================================================
    # Display camera
    # ========================================================

    cv2.imshow(
        "VisionGuide - AI Navigation",
        annotated_frame
    )


    # ========================================================
    # Quit
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("🛑 VisionGuide stopped.")