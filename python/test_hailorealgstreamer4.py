import cv2
import numpy as np
import pyrealsense2 as rs

from hailo_platform import (
    HEF,
    ConfigureParams,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InputVStreamParams,
    InputVStreams,
    OutputVStreamParams,
    OutputVStreams,
    VDevice,
)

# -------- Hailo setup --------
hef_path = "resources/yolov8s.hef"

params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE

target = VDevice(params=params)
hef = HEF(hef_path)

configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

input_vstream_infos = hef.get_input_vstream_infos()
output_vstream_infos = hef.get_output_vstream_infos()

if not input_vstream_infos:
    raise RuntimeError("No input vstream infos found in HEF!")

if not output_vstream_infos:
    raise RuntimeError("No output vstream infos found in HEF!")

input_info = input_vstream_infos[0]
output_info = output_vstream_infos[0]
input_height, input_width, input_channels = input_info.shape

print(f"Model expects input shape: {input_height}x{input_width}x{input_channels}")

# -------- RealSense setup --------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    with network_group.activate(network_group_params):
        input_vstreams = InputVStreams(network_group, InputVStreamParams.make(network_group))
        output_vstreams = OutputVStreams(network_group, OutputVStreamParams.make(network_group))

        input_vstreams_list = list(input_vstreams)
        output_vstreams_list = list(output_vstreams)

        if not input_vstreams_list:
            raise RuntimeError("No input vstreams available after activation!")

        if not output_vstreams_list:
            raise RuntimeError("No output vstreams available after activation!")

        input_vstream = input_vstreams_list[0]
        output_vstream = output_vstreams_list[0]

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            resized_frame = cv2.resize(frame, (input_width, input_height))
            input_data = np.expand_dims(resized_frame, axis=0)  # Add batch dim if required

            # Send input to device
            input_vstream.send(input_data)

            # Receive output from device
            output_data_list = output_vstream.recv()

            if not output_data_list:
                print("Warning: Received empty output!")
                continue

            output_data = output_data_list[0]  # Get first tensor
            print(f"Received output shape: {output_data.shape}")

            # TODO: Add post-processing here

            # Display original frame
            cv2.imshow('Hailo RealSense', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    pipeline.stop()
    target.release()
    cv2.destroyAllWindows()
    print("Finished.")
