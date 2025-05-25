import pyrealsense2 as rs
import numpy as np
import gi
import time
import cv2

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

HEF_PATH = "resources/yolov8s.hef"  # <-- Change this to your Hailo .hef file
MODEL_WIDTH, MODEL_HEIGHT = 640, 640  # Change if your model expects different input size

def create_gst_pipeline():
    pipeline_str = (
    "appsrc name=src is-live=true block=true format=GST_FORMAT_TIME "
    "caps=video/x-raw,format=BGR,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "videoscale ! "
    f"video/x-raw,format=RGB,width={MODEL_WIDTH},height={MODEL_HEIGHT} ! "
    "queue ! "
    f"hailonet hef-path={HEF_PATH} batch-size=1 vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true ! "
    "queue ! "
    "appsink name=sink emit-signals=true max-buffers=1 drop=true"
    )

    return Gst.parse_launch(pipeline_str)

def main():
    # Setup RealSense camera
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_pipeline.start(rs_config)

    # Setup GStreamer pipeline with Hailo inference
    gst_pipeline = create_gst_pipeline()
    appsrc = gst_pipeline.get_by_name("src")
    appsink = gst_pipeline.get_by_name("sink")
    gst_pipeline.set_state(Gst.State.PLAYING)

    try:
        while True:
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())  # 640x480 BGR frame from camera
            data = frame.tobytes()

            # Push frame into appsrc
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            timestamp = int(time.time() * 1e9)
            buf.pts = buf.dts = timestamp
            buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buf.offset = timestamp

            ret = appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print(f"Error pushing buffer to pipeline: {ret}")
                break

            # Pull inference result from appsink
            sample = appsink.emit("pull-sample")
            if not sample:
                print("No sample received from appsink")
                continue

            buffer = sample.get_buffer()
            # Extract ROI/detections from buffer using hailo (if API available)
            # For demonstration, let's assume hailo SDK provides a helper (you might need to adapt)
            import hailo
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

            # Draw detections on original frame
            scale_x = 640 / MODEL_WIDTH
            scale_y = 480 / MODEL_HEIGHT

            for det in detections:
                label = det.get_label()
                confidence = det.get_confidence()
                x, y, w, h = det.get_bbox()

                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Hailo RealSense Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("User interrupted, stopping...")

    finally:
        gst_pipeline.set_state(Gst.State.NULL)
        rs_pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
