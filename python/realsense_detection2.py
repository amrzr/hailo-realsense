import pyrealsense2 as rs
import numpy as np
import cv2
import hailo

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class:
    def __init__(self):
        self.count = 0
        self.use_frame = True
        self.frame = None
        self.new_variable = 42

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count

    def new_function(self):
        return "The meaning of life is:"

    def set_frame(self, frame):
        self.frame = frame


def main():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # User data
    user_data = user_app_callback_class()

    try:
        while True:
            # Wait for a frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            user_data.increment()
            frame = np.asanyarray(color_frame.get_data())
            string_to_print = f"Frame count: {user_data.get_count()}\n"

            detection_count = 0

            # NOTE: Hypothetical direct frame-based Hailo call
            roi = hailo.get_roi_from_buffer(frame)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

            for detection in detections:
                label = detection.get_label()
                bbox = detection.get_bbox()
                confidence = detection.get_confidence()
                if label == "person":
                    track_id = 0
                    track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                    if len(track) == 1:
                        track_id = track[0].get_id()
                    string_to_print += f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
                    detection_count += 1
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add info overlay
            cv2.putText(frame, f"Detections: {detection_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            user_data.set_frame(frame)

            # Show frame
            cv2.imshow("RealSense Detection", frame)
            print(string_to_print)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
