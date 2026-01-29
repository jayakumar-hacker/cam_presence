"""
Face Matcher Module
Matches face embeddings against database
Identity resolution using cosine similarity
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.spatial.distance import cosine
import time

import config
from database.db import get_db


class FaceMatcher:
    """
    Face recognition matcher
    Matches embeddings against stored database of known faces
    Uses cosine similarity for matching
    """
    
    def __init__(self):
        self.db = get_db()
        self.known_embeddings = []
        self.known_identities = []
        self.last_update = None
        
        # Cache management
        self.cache_timeout = 60  # seconds
        
        print("[FaceMatcher] Initialized")
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all known face embeddings from database"""
        print("[FaceMatcher] Loading known faces from database...")
        
        try:
            embeddings_data = self.db.get_all_embeddings()
            
            self.known_embeddings = []
            self.known_identities = []
            
            for data in embeddings_data:
                embedding = data['embedding_vector']
                identity = {
                    'student_id': data['student_id'],
                    'register_number': data['register_number'],
                    'name': data['name'],
                    'quality': data['embedding_quality']
                }
                
                self.known_embeddings.append(embedding)
                self.known_identities.append(identity)
            
            self.last_update = time.time()
            
            print(f"[FaceMatcher] Loaded {len(self.known_embeddings)} face embeddings")
            print(f"[FaceMatcher] Total unique identities: {len(set(i['student_id'] for i in self.known_identities))}")
            
        except Exception as e:
            print(f"[FaceMatcher] Error loading known faces: {e}")
    
    def refresh_cache(self):
        """Refresh cache if timeout expired"""
        if self.last_update is None:
            self.load_known_faces()
            return
        
        elapsed = time.time() - self.last_update
        if elapsed > self.cache_timeout:
            print("[FaceMatcher] Cache timeout, reloading...")
            self.load_known_faces()
    
    def match_face(self, query_embedding: np.ndarray, 
                   top_k: int = 3) -> List[Dict]:
        """
        Match a face embedding against known faces
        
        Args:
            query_embedding: Query face embedding vector
            top_k: Return top K matches
        
        Returns:
            List of matches with similarity scores
        """
        if query_embedding is None:
            return []
        
        if len(self.known_embeddings) == 0:
            print("[FaceMatcher] No known faces in database")
            return []
        
        # Refresh cache if needed
        self.refresh_cache()
        
        # Calculate similarities with all known faces
        similarities = []
        
        for i, known_embedding in enumerate(self.known_embeddings):
            similarity = self._calculate_similarity(query_embedding, known_embedding)
            
            match_data = {
                'identity': self.known_identities[i],
                'similarity': similarity,
                'distance': 1 - similarity
            }
            
            similarities.append(match_data)
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top K matches
        return similarities[:top_k]
    
    def recognize_face(self, query_embedding: np.ndarray) -> Optional[Dict]:
        """
        Recognize a face (best match above threshold)
        
        Args:
            query_embedding: Query face embedding vector
        
        Returns:
            Recognition result or None if no match
        """
        matches = self.match_face(query_embedding, top_k=1)
        
        if not matches:
            return None
        
        best_match = matches[0]
        similarity = best_match['similarity']
        
        # Apply threshold
        if similarity < config.RECOGNITION_THRESHOLD:
            # Unknown face
            return {
                'recognized': False,
                'identity': None,
                'similarity': similarity,
                'status': 'unknown'
            }
        
        if similarity > config.UNKNOWN_THRESHOLD:
            # Very low confidence, likely unknown
            return {
                'recognized': False,
                'identity': None,
                'similarity': similarity,
                'status': 'uncertain'
            }
        
        # Recognized
        return {
            'recognized': True,
            'identity': best_match['identity'],
            'similarity': similarity,
            'status': 'recognized'
        }
    
    def _calculate_similarity(self, embedding1: np.ndarray, 
                             embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = 1 - cosine(embedding1, embedding2)
            
            # Clip to [0, 1]
            similarity = np.clip(similarity, 0, 1)
            
            return float(similarity)
            
        except Exception as e:
            print(f"[FaceMatcher] Similarity calculation error: {e}")
            return 0.0
    
    def batch_recognize(self, embeddings: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        Recognize multiple faces
        
        Args:
            embeddings: List of face embeddings
        
        Returns:
            List of recognition results
        """
        results = []
        
        for embedding in embeddings:
            result = self.recognize_face(embedding)
            results.append(result)
        
        return results
    
    def verify_face(self, embedding1: np.ndarray, 
                   embedding2: np.ndarray) -> Tuple[bool, float]:
        """
        Verify if two embeddings belong to same person
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            (is_same_person, similarity_score)
        """
        similarity = self._calculate_similarity(embedding1, embedding2)
        is_same = similarity >= config.RECOGNITION_THRESHOLD
        
        return is_same, similarity
    
    def add_face_to_database(self, student_id: int, 
                            embedding: np.ndarray,
                            quality: float,
                            camera_id: str = None):
        """
        Add new face embedding to database
        
        Args:
            student_id: Student ID
            embedding: Face embedding vector
            quality: Embedding quality score
            camera_id: Camera that captured the face
        """
        try:
            # Store in database
            self.db.store_embedding(student_id, embedding, quality, camera_id)
            
            print(f"[FaceMatcher] Added new embedding for student {student_id}")
            
            # Refresh cache
            self.load_known_faces()
            
        except Exception as e:
            print(f"[FaceMatcher] Error adding face to database: {e}")
    
    def get_identity_info(self, register_number: str) -> Optional[Dict]:
        """
        Get identity information by register number
        
        Args:
            register_number: Student register number
        
        Returns:
            Identity information or None
        """
        student = self.db.get_student_by_register_number(register_number)
        return student
    
    def get_statistics(self) -> Dict:
        """Get matcher statistics"""
        unique_students = len(set(i['student_id'] for i in self.known_identities))
        
        return {
            'total_embeddings': len(self.known_embeddings),
            'unique_identities': unique_students,
            'avg_embeddings_per_identity': (
                len(self.known_embeddings) / unique_students 
                if unique_students > 0 else 0
            ),
            'last_cache_update': self.last_update
        }


class MultiMatcher:
    """
    Advanced matcher with multiple matching strategies
    Combines multiple embeddings per identity for better accuracy
    """
    
    def __init__(self):
        self.base_matcher = FaceMatcher()
        print("[MultiMatcher] Initialized")
    
    def match_with_voting(self, query_embedding: np.ndarray, 
                         min_votes: int = 2) -> Optional[Dict]:
        """
        Match using voting mechanism across multiple stored embeddings
        
        Args:
            query_embedding: Query embedding
            min_votes: Minimum votes needed for recognition
        
        Returns:
            Recognition result or None
        """
        # Get top matches
        matches = self.base_matcher.match_face(query_embedding, top_k=10)
        
        if not matches:
            return None
        
        # Count votes per identity
        votes = {}
        for match in matches:
            if match['similarity'] >= config.RECOGNITION_THRESHOLD:
                student_id = match['identity']['student_id']
                
                if student_id not in votes:
                    votes[student_id] = {
                        'count': 0,
                        'total_similarity': 0,
                        'identity': match['identity']
                    }
                
                votes[student_id]['count'] += 1
                votes[student_id]['total_similarity'] += match['similarity']
        
        if not votes:
            return None
        
        # Find identity with most votes
        best_identity = max(votes.items(), key=lambda x: (x[1]['count'], x[1]['total_similarity']))
        student_id, vote_data = best_identity
        
        if vote_data['count'] < min_votes:
            return None
        
        avg_similarity = vote_data['total_similarity'] / vote_data['count']
        
        return {
            'recognized': True,
            'identity': vote_data['identity'],
            'similarity': avg_similarity,
            'votes': vote_data['count'],
            'status': 'recognized'
        }
