import pyrealsense2 as rs
import numpy as np
import cv2
import gi
import os
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo
from hailo_apps_infra.hailo_rpi_common import get_caps_from_pad, get_numpy_from_buffer, app_callback_class
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

Gst.init(None)

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42

    def new_function(self):
        return "The meaning of life is:"

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)
    frame = None

    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    person_count = 0
    for det in detections:
        if det.get_label() == "person":
            person_count += 1

    if frame is not None:
        cv2.putText(frame, f"Persons: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        user_data.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Frame {user_data.get_count()} | Persons Detected: {person_count}")
    return Gst.PadProbeReturn.OK

def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

if __name__ == "__main__":
    # ✅ Point to your YAML pipeline config
    os.environ["HAILO_APPS_YAML_PATH"] = "realsense_pipeline.yaml"

    realsense_pipeline = setup_realsense()
    user_data = user_app_callback_class()

    # 🔧 Now you can use the class without any extra arguments
    app = GStreamerDetectionApp(app_callback, user_data)

    import threading
    gst_thread = threading.Thread(target=app.run, daemon=True)
    gst_thread.start()

    while True:
        frames = realsense_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        app.feed_frame(frame)

        output_frame = user_data.get_frame()
        if output_frame is not None:
            cv2.imshow("Hailo + RealSense", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    realsense_pipeline.stop()
    cv2.destroyAllWindows()
