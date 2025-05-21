import dataclasses
import os
import threading
import time
import uuid
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple

# --- Dataclasses ---
@dataclasses.dataclass
class MediaItem:
    id: str
    path: str
    filename: str
    thumbnail: Optional[np.ndarray] = None # Simulate as small numpy array

@dataclasses.dataclass
class TargetMediaItem(MediaItem):
    media_type: str = "image" # "image" or "video"
    duration_seconds: Optional[float] = None # For video
    frame_rate: Optional[float] = None # For video
    resolution: Optional[Tuple[int, int]] = None # (width, height)

@dataclasses.dataclass
class InputFaceItem(MediaItem):
    embedding: Optional[np.ndarray] = None
    source_image_path: str = "" # Path of the original image this face was detected in
    face_bbox: Optional[Tuple[int,int,int,int]] = None # Bbox of face in source_image_path

# --- Mock Processors (for standalone testing and use by MediaManager) ---
class MockFaceDetectorProcessor:
    def detect_faces(self, frame: np.ndarray, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"MockDetector: detect_faces on frame shape {frame.shape}")
        # Simulate finding one face
        h, w = frame.shape[:2]
        if h > 20 and w > 20: # Ensure frame is somewhat valid
            return [{'bbox': (w//4, h//4, w*3//4, h*3//4), 'score': 0.95, 
                     'landmarks_5pt': (np.random.rand(5,2) * [w,h]).tolist()}]
        return []

class MockFaceLandmarkerProcessor:
    def detect_landmarks(self, frame: np.ndarray, faces_data: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"MockLandmarker: detect_landmarks for {len(faces_data)} faces.")
        for face in faces_data:
            if 'landmarks_68pt' not in face: # If not already present
                 face['landmarks_68pt'] = (np.random.rand(68, 2) * 100).tolist() # Dummy landmarks
        return faces_data

class MockFaceRecognizerProcessor:
    def calculate_embedding(self, frame: np.ndarray, landmarks: List[List[float]], params: Dict[str, Any]) -> Optional[np.ndarray]:
        print(f"MockRecognizer: calculate_embedding with {len(landmarks)} landmarks.")
        if landmarks:
            return np.random.rand(512).astype(np.float32) # Simulate a 512-dim embedding
        return None

class MediaManager:
    def __init__(self,
                 face_detector: MockFaceDetectorProcessor, # Type hint with Mocks for clarity
                 face_landmarker: MockFaceLandmarkerProcessor,
                 face_recognizer: MockFaceRecognizerProcessor):
        
        self.face_detector = face_detector
        self.face_landmarker = face_landmarker
        self.face_recognizer = face_recognizer

        self._target_media_items: List[TargetMediaItem] = []
        self._input_face_items: List[InputFaceItem] = []
        self._embedding_cache: Dict[str, np.ndarray] = {} # Key: image_path, Value: embedding

        self._loading_thread: Optional[threading.Thread] = None
        self._stop_loading_event: threading.Event = threading.Event()

        # Callbacks
        self.on_target_media_updated: Optional[Callable[[List[TargetMediaItem]], None]] = None
        self.on_input_faces_updated: Optional[Callable[[List[InputFaceItem]], None]] = None
        self.on_loading_progress: Optional[Callable[[str, float], None]] = None # Message, Progress (0-1)
        self.on_loading_complete: Optional[Callable[[str], None]] = None # Message

        # Allowed extensions
        self.IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        self.VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
        
        print("MediaManager initialized.")

    def _discover_files(self, paths: List[str], include_subfolders: bool, allowed_extensions: List[str]) -> List[str]:
        discovered_files = []
        for path_item in paths:
            if os.path.isfile(path_item):
                if os.path.splitext(path_item)[1].lower() in allowed_extensions:
                    discovered_files.append(path_item)
            elif os.path.isdir(path_item):
                if include_subfolders:
                    for root, _, files in os.walk(path_item):
                        for file in files:
                            if os.path.splitext(file)[1].lower() in allowed_extensions:
                                discovered_files.append(os.path.join(root, file))
                else:
                    for file in os.listdir(path_item):
                        full_path = os.path.join(path_item, file)
                        if os.path.isfile(full_path) and os.path.splitext(file)[1].lower() in allowed_extensions:
                            discovered_files.append(full_path)
            else:
                print(f"MediaManager: Path '{path_item}' is not a valid file or directory. Skipping.")
        return discovered_files

    def _load_target_media_worker_func(self, paths: List[str], include_subfolders: bool):
        print("MediaManager: Starting to load target media...")
        files_to_process = self._discover_files(paths, include_subfolders, self.IMAGE_EXTENSIONS + self.VIDEO_EXTENSIONS)
        new_items = []
        total_files = len(files_to_process)

        for i, file_path in enumerate(files_to_process):
            if self._stop_loading_event.is_set():
                print("MediaManager: Target media loading stopped by event.")
                break
            
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()
            item_id = str(uuid.uuid4())
            
            # Simulate thumbnail (e.g. 64x64 random noise)
            thumbnail = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
            
            if ext in self.IMAGE_EXTENSIONS:
                # Simulate getting image resolution (e.g. by trying to "load" it)
                sim_resolution = (np.random.randint(300,1920), np.random.randint(200,1080))
                item = TargetMediaItem(id=item_id, path=file_path, filename=filename, thumbnail=thumbnail,
                                       media_type="image", resolution=sim_resolution)
            elif ext in self.VIDEO_EXTENSIONS:
                # Simulate video details
                sim_duration = np.random.uniform(10.0, 300.0)
                sim_fps = np.random.choice([24.0, 25.0, 29.97, 30.0, 60.0])
                sim_resolution = (np.random.randint(300,1920), np.random.randint(200,1080))
                item = TargetMediaItem(id=item_id, path=file_path, filename=filename, thumbnail=thumbnail,
                                       media_type="video", duration_seconds=sim_duration, frame_rate=sim_fps,
                                       resolution=sim_resolution)
            else:
                continue # Should not happen due to _discover_files filter

            new_items.append(item)
            self._target_media_items.append(item) # Append one by one for partial updates
            
            if self.on_loading_progress:
                self.on_loading_progress(f"Loaded target: {filename}", (i + 1) / total_files)
            
            # Simulate some work
            time.sleep(0.05) 

        if self.on_target_media_updated:
            self.on_target_media_updated(self._target_media_items) # Send full list after loop
        if self.on_loading_complete:
            status = "Target media loading stopped." if self._stop_loading_event.is_set() else "Target media loading complete."
            self.on_loading_complete(status)
        print(f"MediaManager: {status} Found {len(new_items)} new target items.")

    def load_target_media_async(self, paths: List[str], include_subfolders: bool):
        if self._loading_thread and self._loading_thread.is_alive():
            print("MediaManager Error: Already loading media. Please wait or stop current loading.")
            return
        self._stop_loading_event.clear()
        self._loading_thread = threading.Thread(target=self._load_target_media_worker_func, args=(paths, include_subfolders))
        self._loading_thread.start()

    def _calculate_face_embedding(self, image_path: str, recalculate: bool) -> Optional[Tuple[str, np.ndarray, Tuple[int,int,int,int]]]:
        if not recalculate and image_path in self._embedding_cache:
            print(f"MediaManager: Using cached embedding for {os.path.basename(image_path)}")
            # This version of cache just stores embedding, not bbox. For full InputFaceItem, we'd need more.
            # For simplicity, we'll re-detect face for bbox even if embedding is cached.
            # A more complex cache could store the entire InputFaceItem or its components.
            # Here, let's assume cache key is just path, and we return embedding. Bbox/landmarks are "re-detected".
            
        # Simulate image loading
        sim_frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8) # Dummy frame
        
        # 1. Detect face
        detected_faces = self.face_detector.detect_faces(sim_frame, params={}) # Empty params for mock
        if not detected_faces:
            print(f"MediaManager: No faces detected in {os.path.basename(image_path)}")
            return None
        
        primary_face = detected_faces[0] # Use the first detected face
        face_bbox = primary_face['bbox']

        if not recalculate and image_path in self._embedding_cache:
             return image_path, self._embedding_cache[image_path], face_bbox # Return cached embedding but new bbox


        # 2. Get landmarks (optional for some recognizers, but good practice)
        # face_landmarker might update primary_face dict with landmarks
        self.face_landmarker.detect_landmarks(sim_frame, [primary_face], params={})
        landmarks = primary_face.get('landmarks_5pt') or primary_face.get('landmarks_68pt')
        if not landmarks:
            print(f"MediaManager: No landmarks found for face in {os.path.basename(image_path)}. Proceeding without.")

        # 3. Calculate embedding
        embedding = self.face_recognizer.calculate_embedding(sim_frame, landmarks if landmarks else [], params={})
        if embedding is not None:
            self._embedding_cache[image_path] = embedding
            print(f"MediaManager: Calculated and cached new embedding for {os.path.basename(image_path)}")
            return image_path, embedding, face_bbox
        else:
            print(f"MediaManager: Failed to calculate embedding for {os.path.basename(image_path)}")
            return None


    def _load_input_faces_worker_func(self, paths: List[str], include_subfolders: bool, recalculate_embeddings: bool):
        print("MediaManager: Starting to load input faces...")
        image_files = self._discover_files(paths, include_subfolders, self.IMAGE_EXTENSIONS)
        new_items = []
        total_files = len(image_files)

        for i, file_path in enumerate(image_files):
            if self._stop_loading_event.is_set():
                print("MediaManager: Input faces loading stopped by event.")
                break
            
            filename = os.path.basename(file_path)
            embedding_data = self._calculate_face_embedding(file_path, recalculate_embeddings)

            if embedding_data:
                img_path, embedding, face_bbox = embedding_data
                item_id = str(uuid.uuid4()) # Unique ID for this face item
                # Simulate thumbnail of the face (e.g., crop from a dummy frame)
                thumbnail = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)

                item = InputFaceItem(id=item_id, path=file_path, filename=filename, thumbnail=thumbnail,
                                     embedding=embedding, source_image_path=img_path, face_bbox=face_bbox)
                new_items.append(item)
                self._input_face_items.append(item) # Append one by one
            
            if self.on_loading_progress:
                self.on_loading_progress(f"Processed input face: {filename}", (i + 1) / total_files)
            
            time.sleep(0.1) # Simulate more work for embedding calculation

        if self.on_input_faces_updated:
            self.on_input_faces_updated(self._input_face_items)
        if self.on_loading_complete:
            status = "Input faces loading stopped." if self._stop_loading_event.is_set() else "Input faces loading complete."
            self.on_loading_complete(status)
        print(f"MediaManager: {status} Found {len(new_items)} new input faces.")


    def load_input_faces_async(self, paths: List[str], include_subfolders: bool, recalculate_embeddings: bool):
        if self._loading_thread and self._loading_thread.is_alive():
            print("MediaManager Error: Already loading media. Please wait or stop current loading.")
            return
        self._stop_loading_event.clear()
        self._loading_thread = threading.Thread(target=self._load_input_faces_worker_func, 
                                                args=(paths, include_subfolders, recalculate_embeddings))
        self._loading_thread.start()

    def get_target_media_items(self) -> List[TargetMediaItem]:
        return list(self._target_media_items)

    def get_input_face_items(self) -> List[InputFaceItem]:
        return list(self._input_face_items)

    def get_face_embedding_by_id(self, face_id: str) -> Optional[np.ndarray]:
        for item in self._input_face_items:
            if item.id == face_id:
                return item.embedding
        # Fallback: check cache by path if face_id might be a path (not ideal)
        if face_id in self._embedding_cache: return self._embedding_cache[face_id]
        return None

    def clear_all_media(self):
        print("MediaManager: Clearing all media items and embedding cache.")
        self.stop_loading_media() # Ensure any ongoing loading is stopped
        self._target_media_items.clear()
        self._input_face_items.clear()
        self._embedding_cache.clear()
        if self.on_target_media_updated:
            self.on_target_media_updated(self._target_media_items)
        if self.on_input_faces_updated:
            self.on_input_faces_updated(self._input_face_items)
        print("MediaManager: All media cleared.")

    def stop_loading_media(self):
        if self._loading_thread and self._loading_thread.is_alive():
            print("MediaManager: Setting stop event for loading thread.")
            self._stop_loading_event.set()
            self._loading_thread.join(timeout=3.0) # Wait for thread to finish
            if self._loading_thread.is_alive():
                print("MediaManager Warning: Loading thread did not stop in time.")
            self._loading_thread = None
        else:
            print("MediaManager: No active loading thread to stop.")
        self._stop_loading_event.clear() # Clear for next potential load

if __name__ == '__main__':
    print("--- MediaManager Example Usage ---")

    # 1. Setup mock processors
    mock_detector = MockFaceDetectorProcessor()
    mock_landmarker = MockFaceLandmarkerProcessor()
    mock_recognizer = MockFaceRecognizerProcessor()

    # 2. Instantiate MediaManager
    media_manager = MediaManager(mock_detector, mock_landmarker, mock_recognizer)

    # 3. Define Callbacks
    def handle_targets_updated(items: List[TargetMediaItem]):
        print(f"Callback (Targets Updated): {len(items)} items. First item: {items[0].filename if items else 'N/A'}")
    def handle_faces_updated(items: List[InputFaceItem]):
        print(f"Callback (Faces Updated): {len(items)} items. First item: {items[0].filename if items else 'N/A'} with embedding shape {items[0].embedding.shape if items and items[0].embedding is not None else 'N/A'}")
    def handle_progress(msg: str, progress: float):
        print(f"Callback (Progress): {msg} - {progress*100:.1f}%")
    def handle_complete(msg: str):
        print(f"Callback (Complete): {msg}")

    media_manager.on_target_media_updated = handle_targets_updated
    media_manager.on_input_faces_updated = handle_faces_updated
    media_manager.on_loading_progress = handle_progress
    media_manager.on_loading_complete = handle_complete

    # 4. Create dummy files for testing
    os.makedirs("./dummy_media/subfolder", exist_ok=True)
    dummy_files_info = {
        "./dummy_media/image1.jpg": b"dummy jpg content",
        "./dummy_media/image2.png": b"dummy png content",
        "./dummy_media/video1.mp4": b"dummy mp4 content",
        "./dummy_media/subfolder/image3.jpeg": b"dummy jpeg content",
        "./dummy_media/subfolder/video2.avi": b"dummy avi content",
        "./dummy_media/otherfile.txt": b"not media"
    }
    for path, content in dummy_files_info.items():
        with open(path, "wb") as f:
            f.write(content)
    
    print("\n--- Testing Target Media Loading (includes subfolders) ---")
    media_manager.load_target_media_async(["./dummy_media"], include_subfolders=True)
    time.sleep(1) # Allow loading to proceed

    print("\n--- Testing Input Faces Loading (no subfolders, no recalculate) ---")
    # This will use cached embeddings if any paths overlap, though _calculate_face_embedding currently
    # re-detects bbox. The cache is for the embedding vector itself.
    media_manager.load_input_faces_async(["./dummy_media"], include_subfolders=False, recalculate_embeddings=False)
    time.sleep(1.5) 

    print("\n--- Testing Input Faces Loading (subfolders, with recalculate) ---")
    media_manager.load_input_faces_async(["./dummy_media"], include_subfolders=True, recalculate_embeddings=True)
    time.sleep(1) # Give it time to process a few

    print("\n--- Testing Stop Loading ---")
    media_manager.stop_loading_media() # Should stop the ongoing input face loading
    time.sleep(0.5) # Allow stop to propagate

    print(f"\nTotal target items: {len(media_manager.get_target_media_items())}")
    print(f"Total input face items: {len(media_manager.get_input_face_items())}")
    if media_manager.get_input_face_items():
        face_id_to_get = media_manager.get_input_face_items()[0].id
        retrieved_embedding = media_manager.get_face_embedding_by_id(face_id_to_get)
        print(f"Retrieved embedding for face ID {face_id_to_get} has shape: {retrieved_embedding.shape if retrieved_embedding is not None else 'Not found'}")

    print("\n--- Testing Clear All Media ---")
    media_manager.clear_all_media()
    time.sleep(0.1)
    assert len(media_manager.get_target_media_items()) == 0
    assert len(media_manager.get_input_face_items()) == 0
    assert not media_manager._embedding_cache
    print("Media cleared successfully.")

    # Clean up dummy files
    import shutil
    shutil.rmtree("./dummy_media")
    print("\nCleaned up dummy media files.")
    print("\n--- MediaManager Example Usage Finished ---")
