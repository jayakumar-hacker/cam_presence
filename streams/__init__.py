"""
Video stream handling module
Manages RTSP/HTTP camera streams
"""

from .rtsp_handler import StreamHandler, MultiStreamManager

__all__ = ['StreamHandler', 'MultiStreamManager']
