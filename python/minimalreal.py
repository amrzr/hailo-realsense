import pyrealsense2 as rs
import numpy as np
import cv2
import gi
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

def create_pipeline():
    pipeline_str = """
        appsrc name=src is-live=true block=true format=GST_FORMAT_TIME caps=video/x-raw,format=BGR,width=640,height=480,framerate=30/1 !
        videoconvert !
        autovideosink
    """
    pipeline = Gst.parse_launch(pipeline_str)
    return pipeline

def main():
    # Setup RealSense
    pipeline_rs = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline_rs.start(config)

    # Setup GStreamer
    gst_pipeline = create_pipeline()
    appsrc = gst_pipeline.get_by_name("src")
    gst_pipeline.set_state(Gst.State.PLAYING)

    try:
        while True:
            frames = pipeline_rs.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())  # 640x480x3, BGR
            data = frame.tobytes()

            # Create GstBuffer
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            timestamp = int(time.time() * 1e9)  # nanoseconds
            buf.pts = buf.dts = timestamp
            buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buf.offset = timestamp

            # Push buffer
            ret = appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print(f"Push buffer error: {ret}")
                break

            # Check for 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Stopping...")
        gst_pipeline.set_state(Gst.State.NULL)
        pipeline_rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
