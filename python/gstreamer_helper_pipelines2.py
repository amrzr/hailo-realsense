import os
import sys
import gi
import threading
import pyrealsense2 as rs
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)
import os

def get_source_type(input_source):
    # This function will return the source type based on the input source
    # return values can be "file", "mipi" or "usb"
    if input_source.startswith("/dev/video"):
        return 'usb'
    elif input_source.startswith("rpi"):
        return 'rpi'
    elif input_source.startswith("libcamera"): # Use libcamerasrc element, not suggested
        return 'libcamera'
    elif input_source.startswith('0x'):
        return 'ximage'
    elif input_source.startswith('realsense'):
    	return 'realsense'
    else:
        return 'file'

def QUEUE(name, max_size_buffers=3, max_size_bytes=0, max_size_time=0, leaky='no'):
    """
    Creates a GStreamer queue element string with the specified parameters.

    Args:
        name (str): The name of the queue element.
        max_size_buffers (int, optional): The maximum number of buffers that the queue can hold. Defaults to 3.
        max_size_bytes (int, optional): The maximum size in bytes that the queue can hold. Defaults to 0 (unlimited).
        max_size_time (int, optional): The maximum size in time that the queue can hold. Defaults to 0 (unlimited).
        leaky (str, optional): The leaky type of the queue. Can be 'no', 'upstream', or 'downstream'. Defaults to 'no'.

    Returns:
        str: A string representing the GStreamer queue element with the specified parameters.
    """
    q_string = f'queue name={name} leaky={leaky} max-size-buffers={max_size_buffers} max-size-bytes={max_size_bytes} max-size-time={max_size_time} '
    return q_string

def get_camera_resulotion(video_width=640, video_height=640):
    # This function will return a standard camera resolution based on the video resolution required
    # Standard resolutions are 640x480, 1280x720, 1920x1080, 3840x2160
    # If the required resolution is not standard, it will return the closest standard resolution
    if video_width <= 640 and video_height <= 480:
        return 640, 480
    elif video_width <= 1280 and video_height <= 720:
        return 1280, 720
    elif video_width <= 1920 and video_height <= 1080:
        return 1920, 1080
    else:
        return 3840, 2160


def SOURCE_PIPELINE(video_source, video_width=640, video_height=640, video_format='RGB', name='source', no_webcam_compression=False):
    """
    Creates a GStreamer pipeline string for the video source.

    Args:
        video_source (str): The path or device name of the video source.
        video_width (int, optional): The width of the video. Defaults to 640.
        video_height (int, optional): The height of the video. Defaults to 640.
        video_format (str, optional): The video format. Defaults to 'RGB'.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'source'.

    Returns:
        str: A string representing the GStreamer pipeline for the video source.
    """
    source_type = get_source_type(video_source)

    if source_type == 'usb':
        if no_webcam_compression:
            # When using uncomressed format, only low resolution is supported
            source_element = (
                f'v4l2src device={video_source} name={name} ! '
                f'video/x-raw, width=640, height=480 ! '
                'videoflip name=videoflip video-direction=horiz ! '
            )
        else:
            # Use compressed format for webcam
            width, height = get_camera_resulotion(video_width, video_height)
            source_element = (
                f'v4l2src device={video_source} name={name} ! image/jpeg, framerate=30/1, width={width}, height={height} ! '
                f'{QUEUE(name=f"{name}_queue_decode")} ! '
                f'decodebin name={name}_decodebin ! '
                f'videoflip name=videoflip video-direction=horiz ! '
            )
    elif source_type == 'rpi':
        source_element = (
            f'appsrc name=app_source is-live=true leaky-type=downstream max-buffers=3 ! '
            'videoflip name=videoflip video-direction=horiz ! '
            f'video/x-raw, format={video_format}, width={video_width}, height={video_height} ! '
        )
    elif source_type == 'libcamera':
        source_element = (
            f'libcamerasrc name={name} ! '
            f'video/x-raw, format={video_format}, width=1536, height=864 ! '
        )
    elif source_type == 'ximage':
        source_element = (
            f'ximagesrc xid={video_source} ! '
            f'{QUEUE(name=f"{name}queue_scale_")} ! '
            f'videoscale ! '
        )
    elif source_type == 'realsense':
        source_element = (
            'appsrc name=appsrc is-live=true block=true format=GST_FORMAT_TIME '
            'caps=video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! '
        )

    else:
        source_element = (
            f'filesrc location="{video_source}" name={name} ! '
            f'{QUEUE(name=f"{name}_queue_decode")} ! '
            f'decodebin name={name}_decodebin ! '
        )
        
    source_pipeline = (
        f'{source_element} '
        f'{QUEUE(name=f"{name}_scale_q")} ! '
        f'videoscale name={name}_videoscale n-threads=2 ! '
        f'{QUEUE(name=f"{name}_convert_q")} ! '
        f'videoconvert n-threads=3 name={name}_convert qos=false ! '
        f'video/x-raw, pixel-aspect-ratio=1/1, format={video_format}, width={video_width}, height={video_height} '
        )

    return source_pipeline

def DEPTH_PIPELINE(video_source, video_width=640, video_height=640, video_format='RGB', name='source', no_webcam_compression=False):

    depth_source_element = (
        'appsrc name=depthsrc is-live=true block=true format=GST_FORMAT_TIME '
        'caps=video/x-raw,format=GRAY16_LE,width=1280,height=720,framerate=30/1 ! '
        #'queue ! fakesink'
        'shmsink socket-path=/tmp/depth_socket shm-size=10000000 sync=false wait-for-connection=false'
        #'videoconvert ! x264enc tune=zerolatency ! rtph264pay config-interval=1 pt=96 ! '
        #'udpsink host=127.0.0.1 port=5000'
        #'videoconvert ! jpegenc ! rtpjpegpay ! '
        #'queue ! udpsink host=127.0.0.1 port=5000'
        )

    depth_pipline = (
        f'{depth_source_element} '
        f'{QUEUE(name=f"{name}_depth_scale_q")} ! '
        f'videoscale name={name}_depth_videoscale n-threads=2 ! '
        f'{QUEUE(name=f"{name}_depth_convert_q")} ! '
        f'videoconvert n-threads=3 name={name}_depth_convert qos=false ! '
        f'video/x-raw, pixel-aspect-ratio=1/1, format=GRAY16_LE, width={video_width}, height={video_height} '
        )
    return depth_source_element

