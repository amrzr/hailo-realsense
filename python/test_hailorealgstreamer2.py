import pyrealsense2 as rs
import numpy as np
import cv2
from hailo_platform import (
    HEF,
    ConfigureParams,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    VDevice,
    InputVStreamParams,
    InputVStreams,
    OutputVStreamParams,
    OutputVStreams,
)
import time

# Setup RealSense pipeline
def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

# Initialize Hailo device and network group
def setup_hailo(hef_path):
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE

    target = VDevice(params=params)
    hef = HEF(hef_path)

    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]

    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]

    input_height, input_width, input_channels = input_vstream_info.shape

    return target, network_group, input_height, input_width, input_channels

# Main inference loop
def run_inference(network_group, input_height, input_width, input_channels, rs_pipeline):
    num_of_frames = 100  # adjust as needed

    input_vstream_params = InputVStreamParams.make(network_group)
    output_vstream_params = OutputVStreamParams.make(network_group)

    with network_group.activate(network_group.create_params()):
        with InputVStreams(network_group, input_vstream_params) as input_vstreams, \
             OutputVStreams(network_group, output_vstream_params) as output_vstreams:

            input_vstream = input_vstreams[0]
            output_vstream = output_vstreams[0]

            for _ in range(num_of_frames):
                # Capture RealSense frame
                frames = rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                frame_resized = cv2.resize(frame, (input_width, input_height))

                # Prepare input buffer
                input_data = np.expand_dims(frame_resized, axis=0).astype(input_vstream.dtype)

                # Send to Hailo
                input_vstream.send(input_data)

                # Receive from Hailo
                output_data = output_vstream.recv()

                # === Post-process outputs here ===
                # For now, just print the raw output shape
                print(f"Received output shape: {output_data.shape}")

                # OPTIONAL: Draw dummy box for testing
                cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
                cv2.putText(frame, "Dummy Box", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Show on screen
                cv2.imshow("Hailo RealSense Output", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    HEF_PATH = "/resources/yolov8s.hef"  # <-- replace with your HEF file path

    rs_pipeline = setup_realsense()
    target, network_group, input_height, input_width, input_channels = setup_hailo(HEF_PATH)

    try:
        run_inference(network_group, input_height, input_width, input_channels, rs_pipeline)
    finally:
        rs_pipeline.stop()
        target.release()
        cv2.destroyAllWindows()
