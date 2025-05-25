import pyrealsense2 as rs
import numpy as np
import gi
import time
import cv2
from hailo_platform.pyhailort import pyhailort as hailo

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

HEF_PATH = "resources/yolov8s.hef"  # <-- Update this!

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
    # Use Hailo SDK to get model input size
    with hailo.Device() as device:
        #network_group = device.network_groups.create_from_file(HEF_PATH)
	
        network_group.load()
        input_tensor = network_group.get_input_tensors()[0]
        _, _, model_height, model_width = input_tensor.shape  # NCHW

    print(f"Model input size: {model_width}x{model_height}")

    # Setup RealSense
    pipeline_rs = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline_rs.start(config)

    # Setup GStreamer with Hailo inference
    gst_pipeline = create_pipeline(model_width, model_height)
    appsrc = gst_pipeline.get_by_name("src")
    appsink = gst_pipeline.get_by_name("sink")

    gst_pipeline.set_state(Gst.State.PLAYING)

    try:
        while True:
            frames = pipeline_rs.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())  # 640x480 BGR

            # Resize to model input size
            resized_frame = cv2.resize(frame, (model_width, model_height))
            data = resized_frame.tobytes()

            # Push buffer to appsrc
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            timestamp = int(time.time() * 1e9)
            buf.pts = buf.dts = timestamp
            buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buf.offset = timestamp

            ret = appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print(f"Failed to push buffer: {ret}")
                break

            # Pull inference result from appsink
            sample = appsink.emit("pull-sample")
            if not sample:
                continue

            buffer = sample.get_buffer()
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

            # Draw detection boxes on original frame (640x480)
            for det in detections:
                label = det.get_label()
                confidence = det.get_confidence()
                x, y, w, h = det.get_bbox()  # coords relative to model input size

                scale_x = 640 / model_width
                scale_y = 480 / model_height

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

    finally:
        print("Stopping...")
        gst_pipeline.set_state(Gst.State.NULL)
        pipeline_rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
