import pyrealsense2 as rs
import numpy as np
import cv2
import hailort

# Path to your compiled YOLO model from Hailo
HEF_PATH = "yolov5m.hef"  # or yolov5s.hef, etc.
LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]  # COCO class names

def main():
    # Start RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    with hailort.HailoRT() as rt:
        hef = hailort.Hef(HEF_PATH)
        configured_ng = rt.create_hef_network_group(hef, interface=hailort.HAILO_STREAM_INTERFACE_PCIE)

        input_name = configured_ng.get_input_names()[0]
        output_name = configured_ng.get_output_names()[0]

        input_format = configured_ng.get_input_format(input_name)
        _, c, h_net, w_net = input_format.shape  # NCHW

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
                    h_orig, w_orig, _ = frame.shape

                    # Resize + convert
                    resized = cv2.resize(frame, (w_net, h_net))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)

                    # Send & receive
                    input_vstream.send(input_data)
                    output_data = np.empty(output_vstream.get_frame_size(), dtype=np.float32)
                    output_vstream.receive(output_data)

                    # Output: N x 6 = [x1, y1, x2, y2, conf, class_id]
                    detections = np.reshape(output_data, (-1, 6))

                    # Draw detections
                    for x1, y1, x2, y2, conf, class_id in detections:
                        if conf < 0.5 or int(class_id) >= len(LABELS):
                            continue

                        label = LABELS[int(class_id)]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        # Optionally clamp to frame size
                        x1, x2 = max(0, x1), min(w_orig, x2)
                        y1, y2 = max(0, y1), min(h_orig, y2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow("Hailo YOLO + RealSense", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                pipeline.stop()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
