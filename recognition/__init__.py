"""
Face recognition module
Embedding generation and face matching
"""

from .embedding import FaceEmbedder
from .matcher import FaceMatcher, MultiMatcher

__all__ = ['FaceEmbedder', 'FaceMatcher', 'MultiMatcher']
