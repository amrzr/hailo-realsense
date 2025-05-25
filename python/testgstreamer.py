import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

pipeline_str = """
    videotestsrc pattern=ball ! videoconvert ! autovideosink
"""

pipeline = Gst.parse_launch(pipeline_str)
pipeline.set_state(Gst.State.PLAYING)

try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    pipeline.set_state(Gst.State.NULL)
