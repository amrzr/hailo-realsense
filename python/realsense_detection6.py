import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

import pyrealsense2 as rs
import numpy as np
import cv2
import hailo
import hailo_platform as hp
import time

Gst.init(None)

# Path to your Hailo .hef model file
HEF_PATH = "resources/yolov8s.hef"  # <-- Change this!

def get_model_input_resolution(hef_path):
    with hp.Device() as device:
        #network_group = device.network_groups.create_from_file(hef_path) 
        hef = hp.Hef(hef_path)
        network_group = device.create_network_group(hef)
        network_group.load()
        input_tensor = network_group.get_input_tensors()[0]
        _, _, height, width = input_tensor.shape  # NCHW
        print(height)
        return width, height

def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def create_pipeline(model_width, model_height):
    pipeline_str = f"""
        appsrc name=src is-live=true block=true format=GST_FORMAT_TIME caps=video/x-raw,format=BGR,width=640,height=480,framerate=30/1 !
        videoconvert !
        videoscale !
        video/x-raw,format=NV12,width={model_width},height={model_height} !
        hailonet hef-path={HEF_PATH} !
        hailofilter !
        appsink name=sink emit-signals=true max-buffers=1 drop=true
    """
    pipeline = Gst.parse_launch(pipeline_str)
    return pipeline

def main():
    model_width, model_height = get_model_input_resolution(HEF_PATH)
    print(f"Model input size: {model_width}x{model_height}")

    rs_pipeline = setup_realsense()
    gst_pipeline = create_pipeline(model_width, model_height)

    appsrc = gst_pipeline.get_by_name("src")
    appsink = gst_pipeline.get_by_name("sink")

    gst_pipeline.set_state(Gst.State.PLAYING)

    try:
        while True:
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Resize frame to model input size before pushing
            frame_resized = cv2.resize(frame, (model_width, model_height))
            data = frame_resized.tobytes()

            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            timestamp = int(time.time() * 1e9)
            buf.pts = buf.dts = timestamp
            buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)

            # Push frame into appsrc
            appsrc.emit("push-buffer", buf)

            # Pull inference result from appsink
            sample = appsink.emit("pull-sample")
            if not sample:
                continue

            buffer = sample.get_buffer()
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

            # Draw detections on original full-res frame
            for det in detections:
                label = det.get_label()
                confidence = det.get_confidence()
                x, y, w, h = det.get_bbox()

                # Scale bbox coords back to 640x480 from model input size
                scale_x = 640 / model_width
                scale_y = 480 / model_height

                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Hailo RealSense", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        gst_pipeline.set_state(Gst.State.NULL)
        rs_pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
