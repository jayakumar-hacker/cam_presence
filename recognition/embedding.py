"""
Face Embedding Module
Generates face embeddings using deep learning models
Converts face images to vector representations
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from keras_facenet import FaceNet
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("[WARNING] keras-facenet not available")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARNING] deepface not available")

import config


class FaceEmbedder:
    """
    Face embedding generator
    Converts face images to fixed-size vector representations
    Supports multiple models: FaceNet, VGGFace, ArcFace
    """
    
    def __init__(self, model: str = config.RECOGNITION_MODEL):
        self.model = model
        self.embedder = None
        self.embedding_size = config.EMBEDDING_SIZE
        
        self._initialize_model()
        
        print(f"[FaceEmbedder] Initialized with model: {self.model}")
        print(f"[FaceEmbedder] Embedding size: {self.embedding_size}")
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        if self.model == "facenet":
            if FACENET_AVAILABLE:
                try:
                    self.embedder = FaceNet()
                    self.embedding_size = 512
                    print("[FaceEmbedder] FaceNet loaded successfully")
                    return
                except Exception as e:
                    print(f"[FaceEmbedder] FaceNet initialization failed: {e}")
        
        # Fallback to DeepFace if available
        if DEEPFACE_AVAILABLE:
            self.model = "deepface"
            self.embedding_size = 2622  # VGG-Face default
            print("[FaceEmbedder] Using DeepFace (VGG-Face) as fallback")
        else:
            # Ultimate fallback: simple feature extraction
            self.model = "simple"
            self.embedding_size = 128
            print("[FaceEmbedder] Using simple feature extractor as fallback")
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding vector for a face image
        
        Args:
            face_image: Face image (RGB or BGR)
        
        Returns:
            Embedding vector or None if failed
        """
        if face_image is None or face_image.size == 0:
            return None
        
        try:
            if self.model == "facenet":
                return self._embed_facenet(face_image)
            elif self.model == "deepface":
                return self._embed_deepface(face_image)
            else:
                return self._embed_simple(face_image)
        
        except Exception as e:
            print(f"[FaceEmbedder] Embedding generation error: {e}")
            return None
    
    def _embed_facenet(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding using FaceNet"""
        try:
            # Preprocess face image
            face_image = self._preprocess_face(face_image, target_size=(160, 160))
            
            if face_image is None:
                return None
            
            # Convert BGR to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Expand dimensions for batch processing
            face_image = np.expand_dims(face_image, axis=0)
            
            # Generate embedding
            embedding = self.embedder.embeddings(face_image)
            
            # Normalize
            embedding = embedding[0]  # Get first (and only) embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"[FaceEmbedder] FaceNet embedding error: {e}")
            return None
    
    def _embed_deepface(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding using DeepFace"""
        try:
            # Preprocess face
            face_image = self._preprocess_face(face_image, target_size=(224, 224))
            
            if face_image is None:
                return None
            
            # Convert BGR to RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Generate embedding using VGG-Face
            embedding_objs = DeepFace.represent(
                img_path=face_image,
                model_name="VGG-Face",
                enforce_detection=False
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]["embedding"])
                
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
            
            return None
            
        except Exception as e:
            print(f"[FaceEmbedder] DeepFace embedding error: {e}")
            return None
    
    def _embed_simple(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Simple feature extraction fallback
        Uses histogram and basic features
        """
        try:
            # Resize to standard size
            face_image = self._preprocess_face(face_image, target_size=(64, 64))
            
            if face_image is None:
                return None
            
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Extract features
            # 1. Histogram (64 bins)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = hist.flatten()
            hist = hist / np.sum(hist)  # Normalize
            
            # 2. Resized pixel values (64 values)
            resized = cv2.resize(gray, (8, 8))
            pixels = resized.flatten() / 255.0
            
            # Combine features
            embedding = np.concatenate([hist, pixels])
            
            # Pad or truncate to embedding_size
            if len(embedding) < self.embedding_size:
                embedding = np.pad(embedding, (0, self.embedding_size - len(embedding)))
            else:
                embedding = embedding[:self.embedding_size]
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"[FaceEmbedder] Simple embedding error: {e}")
            return None
    
    def _preprocess_face(self, face_image: np.ndarray, 
                        target_size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
        """
        Preprocess face image for embedding generation
        
        Args:
            face_image: Input face image
            target_size: Target size (width, height)
        
        Returns:
            Preprocessed image or None
        """
        try:
            if face_image is None or face_image.size == 0:
                return None
            
            # Resize to target size
            face_image = cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            if face_image.dtype == np.uint8:
                face_image = face_image.astype(np.float32) / 255.0
            
            return face_image
            
        except Exception as e:
            print(f"[FaceEmbedder] Preprocessing error: {e}")
            return None
    
    def calculate_quality(self, face_image: np.ndarray) -> float:
        """
        Calculate face image quality score
        
        Args:
            face_image: Input face image
        
        Returns:
            Quality score (0-1)
        """
        try:
            if face_image is None or face_image.size == 0:
                return 0.0
            
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Normalize sharpness to 0-1 range (empirical threshold: 500)
            quality = min(sharpness / 500.0, 1.0)
            
            # Calculate brightness
            brightness = np.mean(gray) / 255.0
            
            # Penalize too dark or too bright images
            if brightness < 0.3 or brightness > 0.8:
                quality *= 0.7
            
            return float(quality)
            
        except Exception as e:
            print(f"[FaceEmbedder] Quality calculation error: {e}")
            return 0.0
    
    def batch_generate_embeddings(self, face_images: list) -> list:
        """
        Generate embeddings for multiple faces
        
        Args:
            face_images: List of face images
        
        Returns:
            List of embeddings (None for failed generations)
        """
        embeddings = []
        
        for face_image in face_images:
            embedding = self.generate_embedding(face_image)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model': self.model,
            'embedding_size': self.embedding_size,
            'available_models': {
                'facenet': FACENET_AVAILABLE,
                'deepface': DEEPFACE_AVAILABLE
            }
        }


# Test function
def test_embedder():
    """Test the face embedder"""
    embedder = FaceEmbedder()
    
    # Create a dummy face image
    dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Generate embedding
    embedding = embedder.generate_embedding(dummy_face)
    
    if embedding is not None:
        print(f"[Test] Embedding shape: {embedding.shape}")
        print(f"[Test] Embedding norm: {np.linalg.norm(embedding):.4f}")
        print("[Test] Embedder working correctly!")
    else:
        print("[Test] Embedding generation failed")
    
    # Test quality calculation
    quality = embedder.calculate_quality(dummy_face)
    print(f"[Test] Image quality: {quality:.4f}")


if __name__ == "__main__":
    test_embedder()
