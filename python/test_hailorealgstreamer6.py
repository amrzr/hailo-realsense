import pyrealsense2 as rs
import numpy as np
import gi
import cv2
import time

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

MODEL_WIDTH = 640
MODEL_HEIGHT = 640
HEF_PATH = "resources/yolov8s.hef"
HAILO_FILTER_SO_PATH = "/root/hailo-rpi5-examples/venv_hailo_rpi5_examples/lib/python3.11/site-packages/hailo_apps_infra/../resources/libyolo_hailortpp_postprocess.so"

pipeline_str = (
    "appsrc name=src is-live=true block=true format=GST_FORMAT_TIME "
    "caps=video/x-raw,format=BGR,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "videoscale ! "
    f"video/x-raw,format=RGB,width={MODEL_WIDTH},height={MODEL_HEIGHT} ! "
    "queue ! "
    f"hailonet hef-path={HEF_PATH} batch-size=1 vdevice-group-id=1 "
    "nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true ! "
    "queue ! "
    f"hailofilter so-path={HAILO_FILTER_SO_PATH} function-name=filter_letterbox ! "
    "queue ! "
    "appsink name=sink emit-signals=true max-buffers=1 drop=true"
)

def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            # Example: raw output bytes from hailofilter
            data = map_info.data
            # Process the raw output bytes (e.g., parse detections here)
            print(f"Received inference output buffer with size: {len(data)} bytes")
            buf.unmap(map_info)
    return Gst.FlowReturn.OK

def main():
    # Initialize RealSense pipeline
    pipeline_rs = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline_rs.start(config)

    # Create GStreamer pipeline
    gst_pipeline = Gst.parse_launch(pipeline_str)
    appsrc = gst_pipeline.get_by_name("src")
    appsink = gst_pipeline.get_by_name("sink")

    # Connect signal to handle output from appsink
    appsink.connect("new-sample", on_new_sample)

    gst_pipeline.set_state(Gst.State.PLAYING)

    try:
        while True:
            frames = pipeline_rs.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("No color frame received")
                continue

            frame = np.asanyarray(color_frame.get_data())  # shape (480,640,3), BGR

            # Push frame to appsrc
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            timestamp = int(time.time() * 1e9)
            print(timestamp)
            buf.pts = buf.dts = timestamp
            buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buf.offset = timestamp

            ret = appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print(f"Error pushing buffer: {ret}")
                break

            # Display the camera frame locally (optional)
            cv2.imshow("RealSense Color", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Stopping pipelines...")
        gst_pipeline.set_state(Gst.State.NULL)
        pipeline_rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
