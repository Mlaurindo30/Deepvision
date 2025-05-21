import json
import os
import time # For __main__ example to wait for async loads
from typing import Dict, Any, List, Optional, Callable

# Attempt to import real components; fall back to mocks for standalone testing
try:
    from src.parameter_manager import ParameterManager, SIMULATED_SWAPPER_LAYOUT_DATA
    from src.media_manager import MediaManager, TargetMediaItem, InputFaceItem
    # For MediaManager's dependencies in __main__
    from src.media_manager import MockFaceDetectorProcessor, MockFaceLandmarkerProcessor, MockFaceRecognizerProcessor
    REAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all real components for WorkspaceManager: {e}. Using placeholders.")
    REAL_COMPONENTS_AVAILABLE = False

    class ParameterManager:
        def __init__(self): self._params = {}; print("Mock ParameterManager (Workspace).")
        def get_all_current_values(self): return {"mock_param_key": "mock_value", **self._params}
        def set_parameter_value(self, key, value): self._params[key] = value; print(f"MockParamMgr: {key} set to {value}")
        def register_parameters_from_layout_data(self, data): print("MockParamMgr: Registered params from layout.")
    
    SIMULATED_SWAPPER_LAYOUT_DATA = {} # Mock

    class MediaManager:
        def __init__(self, *args): print("Mock MediaManager (Workspace).")
        def get_target_media_items(self) -> List[Any]: return [type("DummyTMI", (), {"path": "dummy_target.png"})()]
        def get_input_face_items(self) -> List[Any]: return [type("DummyIFI", (), {"source_image_path": "dummy_input.png"})()]
        def clear_all_media(self): print("MockMediaMgr: Cleared all media.")
        def load_target_media_async(self, paths, include_subfolders): print(f"MockMediaMgr: load_target_media_async for {paths}")
        def load_input_faces_async(self, paths, include_subfolders, recalc): print(f"MockMediaMgr: load_input_faces_async for {paths}")
    
    # Mocks for MediaManager's dependencies if MediaManager itself is real in __main__
    class MockFaceDetectorProcessor: pass
    class MockFaceLandmarkerProcessor: pass
    class MockFaceRecognizerProcessor: pass


class WorkspaceManager:
    def __init__(self,
                 parameter_manager: ParameterManager,
                 media_manager: MediaManager,
                 ui_state_provider: Optional[Callable[[], Dict[str, Any]]] = None):
        
        self.parameter_manager = parameter_manager
        self.media_manager = media_manager
        self.ui_state_provider = ui_state_provider if ui_state_provider else lambda: {} # Default empty UI state

        print("WorkspaceManager initialized.")

    def save_workspace(self, filepath: str) -> bool:
        print(f"WorkspaceManager: Attempting to save workspace to '{filepath}'")
        try:
            # 1. Gather data
            target_media_paths = [item.path for item in self.media_manager.get_target_media_items()]
            # Get unique source image paths from input face items
            input_face_image_paths = sorted(list(set(
                item.source_image_path for item in self.media_manager.get_input_face_items() if item.source_image_path
            )))
            
            parameters = self.parameter_manager.get_all_current_values()
            ui_state = self.ui_state_provider()

            workspace_data = {
                "version": "1.0",
                "target_media_paths": target_media_paths,
                "input_face_image_paths": input_face_image_paths,
                "parameters": parameters,
                "ui_state": ui_state # Example: window positions, selected tabs etc.
            }

            # 2. Serialize to JSON and save
            with open(filepath, 'w') as f:
                json.dump(workspace_data, f, indent=4)
            
            print(f"WorkspaceManager: Workspace saved successfully to '{filepath}'.")
            return True

        except (IOError, TypeError) as e:
            print(f"WorkspaceManager Error: Failed to save workspace to '{filepath}'. Reason: {e}")
            return False
        except Exception as e: # Catch any other unexpected errors
            print(f"WorkspaceManager Error: An unexpected error occurred during save: {e}")
            return False


    def load_workspace(self, filepath: str) -> bool:
        print(f"WorkspaceManager: Attempting to load workspace from '{filepath}'")
        if not os.path.exists(filepath):
            print(f"WorkspaceManager Error: File not found - '{filepath}'")
            return False

        try:
            # 1. Deserialize JSON data
            with open(filepath, 'r') as f:
                workspace_data = json.load(f)

            # 2. Extract data
            # version = workspace_data.get("version", "unknown") # Could be used for compatibility checks
            target_media_paths = workspace_data.get("target_media_paths", [])
            input_face_image_paths = workspace_data.get("input_face_image_paths", [])
            parameters = workspace_data.get("parameters", {})
            ui_state = workspace_data.get("ui_state", {}) # For UI to handle

            # 3. Apply data
            # Clear existing media first
            print("WorkspaceManager: Clearing existing media from MediaManager...")
            self.media_manager.clear_all_media() # This should be synchronous or wait if it involves async stop
            # Give a moment for clear_all_media if it has internal async stops
            # In a real app, clear_all_media might need to confirm completion.
            if hasattr(self.media_manager, '_loading_thread') and self.media_manager._loading_thread is not None:
                if self.media_manager._loading_thread.is_alive():
                    print("WorkspaceManager: Waiting for media clearing to finish...")
                    self.media_manager._loading_thread.join(timeout=2.0) # Wait for clear's potential thread join


            print("WorkspaceManager: Loading target media from workspace...")
            if target_media_paths:
                # For simplicity, assume paths are files, not folders, so include_subfolders=False
                self.media_manager.load_target_media_async(target_media_paths, include_subfolders=False)
            
            print("WorkspaceManager: Loading input faces from workspace...")
            if input_face_image_paths:
                # Recalculate_embeddings=False assumes embeddings are either not needed immediately
                # or will be handled by MediaManager's cache or on-demand calculation.
                # If workspace also stored embeddings, this would be different.
                self.media_manager.load_input_faces_async(input_face_image_paths, include_subfolders=False, recalculate_embeddings=False)

            print("WorkspaceManager: Applying parameters...")
            if parameters:
                for key, value in parameters.items():
                    self.parameter_manager.set_parameter_value(key, value)
            
            # (Optional) Apply UI state - For this simulation, just log it
            print(f"WorkspaceManager: Loaded UI state (to be applied by UI): {ui_state}")
            # In a real app, you might emit a signal or call a UIManager method here:
            # if self.ui_manager_instance: self.ui_manager_instance.apply_ui_state(ui_state)

            print(f"WorkspaceManager: Workspace loaded successfully from '{filepath}'.")
            return True

        except (IOError, json.JSONDecodeError) as e:
            print(f"WorkspaceManager Error: Failed to load workspace from '{filepath}'. Reason: {e}")
            return False
        except Exception as e: # Catch any other unexpected errors
            print(f"WorkspaceManager Error: An unexpected error occurred during load: {e}")
            return False


if __name__ == '__main__':
    print("--- WorkspaceManager Example Usage ---")

    # 1. Instantiate real/mock ParameterManager and MediaManager
    if REAL_COMPONENTS_AVAILABLE:
        param_manager = ParameterManager()
        # For real ParameterManager, register some dummy layout data
        param_manager.register_parameters_from_layout_data(
            SIMULATED_SWAPPER_LAYOUT_DATA or {"test_cat": {"test_param_Slider": {"default": 5}}}
        )
        
        # MediaManager needs its mock processors
        mock_detector = MockFaceDetectorProcessor()
        mock_landmarker = MockFaceLandmarkerProcessor()
        mock_recognizer = MockFaceRecognizerProcessor()
        media_manager = MediaManager(mock_detector, mock_landmarker, mock_recognizer)
    else: # Use full mocks if real components aren't available
        param_manager = ParameterManager() 
        media_manager = MediaManager()


    # 2. Define a mock UI state provider
    def mock_ui_state_provider() -> Dict[str, Any]:
        print("Mock UI State Provider: Providing dummy UI state.")
        return {"window_size": (1280, 720), "active_tab": "processing_tab"}

    # 3. Instantiate WorkspaceManager
    workspace_manager = WorkspaceManager(param_manager, media_manager, mock_ui_state_provider)
    workspace_filepath = "test_workspace.json"

    # --- Test Save ---
    print("\n--- Testing Save Workspace ---")
    # Populate ParameterManager with some values
    param_manager.set_parameter_value("brightness_slider", 0.75)
    param_manager.set_parameter_value("contrast_toggle", True)

    # Populate MediaManager (simulate loading some media so paths are available)
    # Create dummy files for MediaManager to "find"
    os.makedirs("./dummy_ws_media/targets", exist_ok=True)
    os.makedirs("./dummy_ws_media/inputs", exist_ok=True)
    with open("./dummy_ws_media/targets/video1.mp4", "w") as f: f.write("dummy video")
    with open("./dummy_ws_media/inputs/face1.jpg", "w") as f: f.write("dummy face image")
    
    # Use MediaManager's async loading and wait (or use internal lists for direct test)
    print("WorkspaceManager Test: Loading dummy media into MediaManager for save test...")
    if hasattr(media_manager, 'load_target_media_async'): # If it's the real MediaManager
        media_manager.load_target_media_async(["./dummy_ws_media/targets/video1.mp4"], include_subfolders=False)
        media_manager.load_input_faces_async(["./dummy_ws_media/inputs/face1.jpg"], include_subfolders=False, recalculate_embeddings=False)
        
        # Wait for async loading to (hopefully) complete
        # In a real test, you'd use callbacks or join threads properly.
        print("WorkspaceManager Test: Waiting for MediaManager async loads to process a bit...")
        time.sleep(1.0) # Give some time for async operations
        if hasattr(media_manager, '_loading_thread') and media_manager._loading_thread is not None:
            if media_manager._loading_thread.is_alive():
                media_manager.stop_loading_media() # Stop it if still running
    else: # If it's the mock MediaManager, its getters are hardcoded
        print("WorkspaceManager Test: Using Mock MediaManager, paths are hardcoded for save.")


    save_success = workspace_manager.save_workspace(workspace_filepath)
    assert save_success, "Workspace save failed."
    assert os.path.exists(workspace_filepath), "Workspace file was not created."
    print(f"Save test completed. Workspace file: {workspace_filepath}")

    # --- Test Load ---
    print("\n--- Testing Load Workspace ---")
    # Optionally, clear current state of managers before loading to verify effect
    if hasattr(media_manager, 'clear_all_media'): media_manager.clear_all_media()
    param_manager.set_parameter_value("brightness_slider", 0.0) # Reset a param

    load_success = workspace_manager.load_workspace(workspace_filepath)
    assert load_success, "Workspace load failed."
    
    print("\n--- Verifying Loaded State ---")
    # Verify ParameterManager state
    loaded_params = param_manager.get_all_current_values()
    print(f"Parameters after load: {loaded_params}")
    assert loaded_params.get("brightness_slider") == 0.75, "Parameter 'brightness_slider' not loaded correctly."
    assert loaded_params.get("contrast_toggle") == True, "Parameter 'contrast_toggle' not loaded correctly."

    # Verify MediaManager state (e.g. by checking if its load methods were called - logs will show this)
    # If using real MediaManager, its lists would be populated.
    # For this test, the print statements from MediaManager's load_..._async methods (real or mock)
    # would indicate they were called by WorkspaceManager.load_workspace.
    
    # Wait for async loading from workspace load to process a bit
    if hasattr(media_manager, '_loading_thread') and media_manager._loading_thread is not None:
        if media_manager._loading_thread.is_alive():
            print("WorkspaceManager Test: Waiting for MediaManager async loads (from workspace) to process...")
            time.sleep(1.0)
            media_manager.stop_loading_media()

    print("\nLoad test completed.")

    # Clean up
    if os.path.exists(workspace_filepath):
        os.remove(workspace_filepath)
        print(f"Cleaned up workspace file: {workspace_filepath}")
    
    import shutil
    if os.path.exists("./dummy_ws_media"):
        shutil.rmtree("./dummy_ws_media")
        print("Cleaned up dummy_ws_media directory.")

    print("\n--- WorkspaceManager Example Usage Finished ---")
