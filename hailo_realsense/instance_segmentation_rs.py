import sys
sys.path.insert(1, '../python')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import pyrealsense2 as rs

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from instance_segmentation_pipeline_rs import GStreamerInstanceSegmentationApp

depth_frame = None

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_skip = 2  # Process every 2nd frame to reduce compute

# Predefined colors (BGR format)
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0)   # Olive
]

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    global depth_frame
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    #print(buffer)
    #success, map_info = buffer.map(Gst.MapFlags.READ)
    #print(map_info.data)
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Skip frames to reduce compute
    if user_data.get_count() % user_data.frame_skip != 0:
        return Gst.PadProbeReturn.OK

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Reduce the resolution by a factor of 4
    reduced_width = width // 4
    reduced_height = height // 4

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    reduced_frame = None
    #if user_data.use_frame and format is not None and width is not None and height is not None:
    if True:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)
        #print(get_numpy_from_buffer(buffer.get_buffer(), format, width, height))
        reduced_frame = cv2.resize(frame, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    #ret_depth, frame_depth = depth_cap.read()
    #print(frame_depth)
    
    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if True: #label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()

            #string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}")
            print(f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}")
            # Instance segmentation mask from detection (if available)
            if True: #user_data.use_frame:
                masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
                if len(masks) != 0:
                    mask = masks[0]
                    # Note that the mask is a 1D array, you need to reshape it to get the original shape
                    mask_height = mask.get_height()
                    mask_width = mask.get_width()
                    data = np.array(mask.get_data())
                    data = data.reshape((mask_height, mask_width))
                    # Resize the mask to the ROI size
                    roi_width = int(bbox.width() * reduced_width)
                    roi_height = int(bbox.height() * reduced_height)
                    resized_mask_data = cv2.resize(data, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)

                    # Calculate the ROI coordinates
                    x_min, y_min = int(bbox.xmin() * reduced_width), int(bbox.ymin() * reduced_height)
                    x_max, y_max = x_min + roi_width, y_min + roi_height

                    # Ensure the ROI dimensions are within the frame boundaries and handle negative values
                    y_min = max(y_min, 0)
                    x_min = max(x_min, 0)
                    y_max = min(y_max, reduced_frame.shape[0])
                    x_max = min(x_max, reduced_frame.shape[1])
                    print("Centroid = ",4*(y_max-y_min), 4*(x_max-x_min))
                    print("Distance (cm) = ",(depth_frame[4*(y_max-y_min)][4*(x_max-x_min)])/10)
                    """
                    # Ensure ROI dimensions are valid
                    if x_max > x_min and y_max > y_min:
                        # Add mask overlay to the frame
                        mask_overlay = np.zeros_like(reduced_frame)
                        color = COLORS[track_id % len(COLORS)]  # Get color based on track_id
                        mask_overlay[y_min:y_max, x_min:x_max] = (resized_mask_data[:y_max-y_min, :x_max-x_min, np.newaxis] > 0.5) * color
                        reduced_frame = cv2.addWeighted(reduced_frame, 1, mask_overlay, 0.5, 0)
                    """
    print(string_to_print)
    if user_data.use_frame:
        # Convert the frame to BGR
        reduced_frame = cv2.cvtColor(reduced_frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(reduced_frame)

    return Gst.PadProbeReturn.OK

def depth_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Example: log depth frame size
    #print(f"[Depth] Received buffer of size: {buffer.get_size()}")

    # Optional: map buffer memory if you need the raw data
    result, map_info = buffer.map(Gst.MapFlags.READ)
    if result:
        depth_data = map_info.data  # this is the raw bytes
        global depth_frame
        depth_frame = np.frombuffer(depth_data, dtype='<u2').reshape((720,1280))
        #print(depth_frame)
        # process depth_data here...
        buffer.unmap(map_info)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerInstanceSegmentationApp(app_callback, depth_callback, user_data)
    app.run()
