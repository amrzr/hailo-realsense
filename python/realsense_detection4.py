import pyrealsense2 as rs
import numpy as np
import cv2
import hailort
import time

# -----------------------------------------------------------------------------------------------
# User-defined class similar to your app_callback_class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class:
    def __init__(self):
        self.frame_count = 0
        self.new_variable = 42
        self.frame = None
        self.use_frame = True  # mimic your original flag

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

    def set_frame(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame

    def new_function(self):
        return "The meaning of life is:"

# -----------------------------------------------------------------------------------------------
# Detection and processing function replacing app_callback
# -----------------------------------------------------------------------------------------------
def process_frame(frame, user_data, configured_ng, input_vstream, output_vstream):
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Preprocess frame for YOLO input
    input_format = configured_ng.get_input_format(input_vstream.name)
    _, c, h_net, w_net = input_format.shape

    # Resize and convert color (assuming network expects RGB and NCHW format)
    resized = cv2.resize(frame, (w_net, h_net))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to 0-1 float32
    input_tensor = rgb.astype(np.float32) / 255.0
    # Change to CHW
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Copy data to hailo input stream
    input_vstream.write(input_tensor)

    # Read output
    output_tensor = output_vstream.read()
    
    # Get detections from output buffer
    roi = hailort.get_roi_from_buffer(output_tensor)
    detections = roi.get_objects_typed(hailort.HAILO_DETECTION)

    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        if label == "person":
            track_id = 0
            track = detection.get_objects_typed(hailort.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            detection_count += 1

    if user_data.use_frame:
        # Draw detection count and custom message on frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        user_data.set_frame(frame)

    print(string_to_print)

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    user_data = user_app_callback_class()
    hef_path = "yolov5m.hef"  # Update path to your .hef file

    # Configure RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    with hailort.HailoRT() as rt, open(hef_path, 'rb') as hef_file:
        hef = hailort.Hef(hef_file.read())
        configured_ng = rt.create_hef_network_group(hef, interface=hailort.HAILO_STREAM_INTERFACE_PCIE)
        input_name = configured_ng.get_input_names()[0]
        output_name = configured_ng.get_output_names()[0]

        with configured_ng.activate() as activated_ng:
            input_vstream = activated_ng.get_input_vstream(input_name)
            output_vstream = activated_ng.get_output_vstream(output_name)

            try:
                while True:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    frame = np.asanyarray(color_frame.get_data())

                    # Run detection and process frame
                    process_frame(frame, user_data, configured_ng, input_vstream, output_vstream)

                    # Show the frame with overlays
                    if user_data.use_frame:
                        cv2.imshow('Detections', user_data.get_frame())
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            finally:
                pipeline.stop()
                cv2.destroyAllWindows()
