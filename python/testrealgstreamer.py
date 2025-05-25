import pyrealsense2 as rs
import numpy as np
import gi
import time
import cv2

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

def create_pipeline():
    pipeline_str = """
        appsrc name=src is-live=true block=true format=GST_FORMAT_TIME caps=video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 !
        videoconvert !
        xvimagesink sync=false
    """
    pipeline = Gst.parse_launch(pipeline_str)
    return pipeline

def main():
    # Setup RealSense
    pipeline_rs = rs.pipeline()
    print(pipeline_rs)
    config = rs.config()
    print(config)
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found")
    dev = devices[0]
    print(f"Device: {dev.get_info(rs.camera_info.name)}")
    for sensor in dev.query_sensors():
        print(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
        for p in sensor.get_stream_profiles():
            if p.stream_type() == rs.stream.color:
                v = p.as_video_stream_profile()
                print(f"    Color: {v.width}x{v.height} @ {v.fps()} ({v.format()})")
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
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
                print("No color frame received")
                continue

            frame = np.asanyarray(color_frame.get_data())  # 640x480x3 BGR
            data = frame.tobytes()

            print(f"Frame shape: {frame.shape}, Buffer size: {len(data)}")

            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            timestamp = int(time.time() * 1e9)
            buf.pts = buf.dts = timestamp
            buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buf.offset = timestamp

            ret = appsrc.emit("push-buffer", buf)
            print(f"Push buffer return: {ret}")
            if ret != Gst.FlowReturn.OK:
                print(f"Failed to push buffer: {ret}")
                break

            # Check key press with OpenCV window (just a dummy window)
            cv2.imshow("Dummy", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Stopping pipeline and camera...")
        gst_pipeline.set_state(Gst.State.NULL)
        pipeline_rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
