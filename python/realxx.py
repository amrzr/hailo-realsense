import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import cv2
import numpy as np
import setproctitle
import pyrealsense2 as rs
import time

from hailo_apps_infra.hailo_rpi_common import get_default_parser, detect_hailo_arch
from hailo_apps_infra.gstreamer_helper_pipelines import (
    INFERENCE_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)
from hailo_apps_infra.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback


class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_default_parser()
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to custom labels JSON file",
        )

        super().__init__(parser, user_data)

        self.video_width = 640
        self.video_height = 480  # RealSense default

        self.batch_size = 1
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch

        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        elif self.arch == "hailo8":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov6n.hef')
        else:
            self.hef_path = os.path.join(self.current_path, '../resources/yolov6n_h8l.hef')

        self.post_process_so = os.path.join(self.current_path, '../resources/libyolo_hailortpp_postprocess.so')
        self.post_function_name = "filter"
        self.labels_json = self.options_menu.labels_json

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        setproctitle.setproctitle("Hailo RealSense Detection App")

        self.create_pipeline()

    def get_pipeline_string(self):
        # Use appsrc instead of SOURCE_PIPELINE
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str
        )
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        pipeline_string = (
            f'appsrc name=src is-live=true block=true format=GST_FORMAT_TIME caps=video/x-raw,format=BGR,width={self.video_width},height={self.video_height},framerate=30/1 ! '
            f'videoconvert ! videoscale ! video/x-raw,format=NV12,width={self.video_width},height={self.video_height},framerate=30/1 ! '
            f'{detection_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
            )
        print("GStreamer pipeline:")
        print(pipeline_string)
        return pipeline_string

    def run(self):
        rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, self.video_width, self.video_height, rs.format.bgr8, 30)
        rs_pipeline.start(rs_config)

        self.pipeline.set_state(Gst.State.PLAYING)
        appsrc = self.pipeline.get_by_name('src')

        try:
            while True:
                frames = rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())

                data = frame.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                timestamp = int(time.time() * 1e9)
                buf.pts = buf.dts = timestamp
                buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)

                retval = appsrc.emit("push-buffer", buf)
                if retval != Gst.FlowReturn.OK:
                    print("Failed to push buffer to pipeline")
                    break

                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("Exiting on user request")
                    break

        finally:
            self.pipeline.set_state(Gst.State.NULL)
            rs_pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
