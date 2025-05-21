import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable

# Attempt to import real components; fall back to mocks for standalone testing or if some are missing
try:
    from src.settings_manager import SettingsManager
    from src.parameter_manager import ParameterManager, SIMULATED_SWAPPER_LAYOUT_DATA # For param_manager example
    from src.processing_orchestrator import ProcessingOrchestrator
    from src.ai_model_manager import AIModelManager, SIMULATED_MODELS_DATA as ORCH_SIM_MODELS # Orchestrator's models
    # For AIFunctionProcessors, we'll use mocks in __main__ to simplify UIManager example
    REAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all real components for UIManager: {e}. Using placeholders for some.")
    REAL_COMPONENTS_AVAILABLE = False

    class SettingsManager:
        def __init__(self, config_file='config.json'): print(f"Mock SettingsManager initialized (config: {config_file}).")
        def get_setting(self, key): return {"language": "en", "theme": "dark"}.get(key, None)
        def set_setting(self, key, value): print(f"Mock SettingsManager: {key} set to {value}")

    class ParameterManager:
        def __init__(self): self._observers = []; self._params = {}; print("Mock ParameterManager initialized.")
        def set_parameter_value(self, name, value): self._params[name] = value; print(f"Mock ParamManager: {name} set to {value}"); self._notify(name,value)
        def get_all_current_values(self): return {"mock_param": 123, **self._params}
        def subscribe(self, obs): self._observers.append(obs)
        def _notify(self, name, value): [obs(name,value) for obs in self._observers]
        def register_parameters_from_layout_data(self, data): print("Mock ParamManager: Registered params from layout.")
    
    SIMULATED_SWAPPER_LAYOUT_DATA = {} # Placeholder for mock
    ORCH_SIM_MODELS = {} # Placeholder for mock

    class AIModelManager:
        def __init__(self, models_data, initial_device_str="cpu", cache_limit=3): print("Mock AIModelManager used in UIManager example.")
    
    class ProcessingOrchestrator:
        def __init__(self, *args): 
            print("Mock ProcessingOrchestrator used.")
            self.on_frame_processed: Optional[Callable[[Any], None]] = None
            self.on_processing_started: Optional[Callable[[], None]] = None
            self.on_processing_stopped: Optional[Callable[[Optional[str]], None]] = None
            self.on_error: Optional[Callable[[str], None]] = None
            self.on_progress_update: Optional[Callable[[float], None]] = None
        def process_single_image(self, path, params, embeddings): print(f"Mock Orchestrator: process_single_image for {path}"); return "MockFrameData"
        def start_video_processing(self, path, params, embeddings): print(f"Mock Orchestrator: start_video_processing for {path}")
        def stop_processing(self): print("Mock Orchestrator: stop_processing called.")

# --- Mock/Placeholder Classes for UIManager dependencies ---
class MockMediaManager:
    def __init__(self):
        print("MockMediaManager initialized.")
    def get_media_path(self, media_type="image") -> str:
        return f"dummy_{media_type}.png" if media_type=="image" else f"dummy_{media_type}.mp4"

class MockWorkspaceManager:
    def __init__(self):
        print("MockWorkspaceManager initialized.")
    def get_source_embeddings(self) -> Dict[str, Any]: # Any is np.ndarray
        print("MockWorkspaceManager: Providing dummy source embeddings.")
        return {"source_face_1": np.random.rand(512).astype(np.float32)}


class UIManager:
    def __init__(self,
                 settings_manager: SettingsManager,
                 parameter_manager: ParameterManager,
                 processing_orchestrator: ProcessingOrchestrator,
                 media_manager: Any, # Using Any for mocks
                 workspace_manager: Any): # Using Any for mocks
        
        self.settings_manager = settings_manager
        self.parameter_manager = parameter_manager
        self.processing_orchestrator = processing_orchestrator
        self.media_manager = media_manager
        self.workspace_manager = workspace_manager

        self.current_language = self.settings_manager.get_setting("language") or "en"
        self.current_theme = self.settings_manager.get_setting("theme") or "default"
        
        self.translations = {
            "en": {"greeting": "Hello", "start_button": "Start Processing"},
            "pt_BR": {"greeting": "Olá", "start_button": "Iniciar Processamento"}
        }

        self._setup_ui_structure()
        self._connect_ui_signals()

        # Connect orchestrator callbacks
        self.processing_orchestrator.on_frame_processed = self.display_processed_frame
        self.processing_orchestrator.on_processing_started = self.handle_processing_started
        self.processing_orchestrator.on_processing_stopped = self.handle_processing_stopped
        self.processing_orchestrator.on_error = self.display_error_message
        self.processing_orchestrator.on_progress_update = self.update_progress_bar
        
        # Subscribe to parameter changes from ParameterManager
        self.parameter_manager.subscribe(self.handle_parameter_change_from_backend)

        print(f"UIManager initialized. Language: {self.current_language}, Theme: {self.current_theme}")

    def _setup_ui_structure(self):
        print("UIManager: Setting up UI structure (simulated).")
        # In a real app, this would involve creating Qt/Tkinter/Web widgets

    def _connect_ui_signals(self):
        print("UIManager: Connecting UI signals (e.g., button clicks to handlers) (simulated).")
        # E.g., self.ui.startButton.clicked.connect(self.handle_start_processing_click)

    def apply_language(self, lang_code: str):
        self.current_language = lang_code
        self.settings_manager.set_setting("language", lang_code) # Persist
        print(f"UIManager: Language changed to {lang_code}. UI would be updated.")
        # In a real app, trigger UI text refresh here

    def apply_theme(self, theme_name: str):
        self.current_theme = theme_name
        self.settings_manager.set_setting("theme", theme_name) # Persist
        print(f"UIManager: Theme changed to {theme_name}. UI style would be updated.")
        # In a real app, trigger UI style refresh here

    def tr(self, text_key: str) -> str:
        return self.translations.get(self.current_language, {}).get(text_key, text_key)

    # --- Event Handlers (simulating calls from UI elements) ---
    def handle_parameter_widget_changed(self, param_name: str, new_value: Any):
        print(f"UIManager: UI widget for '{param_name}' changed to '{new_value}'. Forwarding to ParameterManager.")
        self.parameter_manager.set_parameter_value(param_name, new_value)

    def handle_parameter_change_from_backend(self, param_name: str, new_value: Any):
        # This method is called by ParameterManager's notification system
        print(f"UIManager: Received backend update for '{param_name}' to '{new_value}'. UI widget would be updated if it exists.")
        # In a real app: self.ui.widgets[param_name].setValue(new_value) (pseudo-code)

    def handle_start_processing_click(self):
        print(f"UIManager: '{self.tr('start_button')}' clicked.")
        
        current_processing_params = self.parameter_manager.get_all_current_values()
        print(f"UIManager: Collected processing parameters: {current_processing_params}")
        
        # Simulate getting media path and source embeddings (using mocks)
        # Assume UI has a way to select media type (image/video)
        simulated_media_type = "image" # or "video"
        media_path = self.media_manager.get_media_path(simulated_media_type)
        source_embeddings = self.workspace_manager.get_source_embeddings()

        print(f"UIManager: Media path: '{media_path}', Source embeddings keys: {list(source_embeddings.keys())}")

        # For this simulation, let's use a simplified set of params for the orchestrator
        # In a real app, these would be structured more carefully based on UI selections
        orchestrator_params = {
            "run_face_detection": current_processing_params.get("enable_detection", True),
            "detector_params": {"score_threshold": current_processing_params.get("detection_threshold", 0.5)},
            # Add other relevant params based on what's in current_processing_params
            "image_resolution_sim": (100,100) # Small for quick test
        }

        if simulated_media_type == "image":
            self.processing_orchestrator.process_single_image(media_path, orchestrator_params, source_embeddings)
        elif simulated_media_type == "video":
            orchestrator_params["total_frames_sim"] = 3 # Short video for test
            orchestrator_params["frame_rate_sim"] = 1.0
            self.processing_orchestrator.start_video_processing(media_path, orchestrator_params, source_embeddings)
        else:
            self.display_error_message(f"Unsupported media type: {simulated_media_type}")


    # --- ProcessingOrchestrator Callback Implementations ---
    def display_processed_frame(self, frame_data: ProcessingOrchestrator.FrameData):
        print(f"UIManager Callback (display_processed_frame): Received data for frame {frame_data.frame_number}.")
        if frame_data.error:
            self.display_error_message(f"Frame {frame_data.frame_number} Error: {frame_data.error}")
        # In a real UI: update image display with frame_data.processed_frame, update mask display with frame_data.mask

    def handle_processing_started(self):
        print("UIManager Callback (handle_processing_started): Processing has started.")
        # In a real UI: disable start button, show progress bar, etc.

    def handle_processing_stopped(self, error_message: Optional[str]):
        if error_message:
            print(f"UIManager Callback (handle_processing_stopped): Processing stopped with error: {error_message}")
        else:
            print("UIManager Callback (handle_processing_stopped): Processing finished successfully.")
        # In a real UI: enable start button, hide progress bar, etc.

    def display_error_message(self, message: str):
        print(f"UIManager Callback (display_error_message): ERROR - {message}")
        # In a real UI: show a message box or status bar error

    def update_progress_bar(self, progress_value: float): # Changed to float
        print(f"UIManager Callback (update_progress_bar): Progress: {progress_value*100:.1f}%")
        # In a real UI: self.ui.progressBar.setValue(int(progress_value * 100))


if __name__ == '__main__':
    print("--- UIManager Example Usage ---")

    # 1. Instantiate real managers (Settings, Parameters)
    # Use real components if available, else mocks defined at top of file will be used
    settings_mgr = SettingsManager() # Uses real if available, else mock
    param_mgr = ParameterManager()    # Uses real if available, else mock
    
    # For ParameterManager, if it's the real one, register some dummy layout data
    if REAL_COMPONENTS_AVAILABLE and hasattr(param_mgr, 'register_parameters_from_layout_data'):
        # SIMULATED_SWAPPER_LAYOUT_DATA would be imported from src.parameter_manager
        # If not, a mock one is fine for this UI test.
        # Let's define a simple one here if the real one isn't available through the import.
        if not SIMULATED_SWAPPER_LAYOUT_DATA: # If using mock ParameterManager, this might be empty
             example_layout_data = {
                "general_settings": {
                    "enable_detection_Toggle": {"widget_type": "Toggle", "label": "Enable Detection", "default": True},
                    "detection_threshold_Slider": {"widget_type": "Slider", "label": "Detection Threshold", "min":0.1, "max":1.0, "step":0.05, "default":0.5}
                }
            }
        else: # Use the one from parameter_manager if available
            example_layout_data = SIMULATED_SWAPPER_LAYOUT_DATA
        
        param_mgr.register_parameters_from_layout_data(example_layout_data)
        print("Registered example layout data with ParameterManager.")


    # 2. Instantiate Mocks for MediaManager, WorkspaceManager
    media_mgr_mock = MockMediaManager()
    workspace_mgr_mock = MockWorkspaceManager()

    # 3. Instantiate AIModelManager and mock AIFunctionProcessors for Orchestrator
    # Use ORCH_SIM_MODELS defined based on imports
    # It will be empty if REAL_COMPONENTS_AVAILABLE is True but ai_model_manager failed, so re-populate
    if REAL_COMPONENTS_AVAILABLE and not ORCH_SIM_MODELS :
         ORCH_SIM_MODELS = { # Ensure models for default orchestrator ops are present
            "RetinaFace": {"path":"dummy.onnx"}, "Landmarker_68pt": {"path":"dummy.onnx"},
            "Inswapper128ArcFace": {"path":"dummy.onnx"}, "Inswapper128": {"path":"dummy.onnx"},
            "LivePortrait_Editor": {"path":"dummy.onnx"}, "Makeup_Model": {"path":"dummy.onnx"},
            "GFPGAN-v1.4": {"path":"dummy.onnx"}, "RealESRGAN_x4plus": {"path":"dummy.onnx"},
            "DFL_XSeg_Face": {"path":"dummy.onnx"}
         }

    ai_model_mgr_orch = AIModelManager(ORCH_SIM_MODELS) # Real or mock based on imports

    # Define simple mock AI processors for the orchestrator example
    class MockAIProcessor:
        def __init__(self, name, model_manager): self.name = name; self.model_manager = model_manager
        def __call__(self, *args, **kwargs): print(f"MockAIProcessor '{self.name}' called."); return args[0] if args else None # Return first arg (frame)
    
    # Need to handle methods with specific return signatures if orchestrator expects them
    class MockFaceDetectorP(MockAIProcessor):
        def detect_faces(self, frame, params): print(f"MockFaceDetectorP.detect_faces called."); return [{'bbox':[10,10,60,60], 'score':0.9, 'landmarks_5pt': (np.random.rand(5,2)*50+10).tolist()}]
    class MockFaceLandmarkerP(MockAIProcessor):
        def detect_landmarks(self, frame, faces, params): print(f"MockFaceLandmarkerP.detect_landmarks called."); [f.update({'landmarks_68pt': (np.random.rand(68,2)*50+10).tolist()}) for f in faces]; return faces
    class MockFaceSwapperP(MockAIProcessor):
        def __init__(self, name, mm): super().__init__(name,mm); self.emap_inswapper128 = np.random.rand(512,512) # Needed by orchestrator example
        def get_source_face_embedding(self, frame, face, params): print(f"MockFaceSwapperP.get_source_embedding called."); return np.random.rand(512).astype(np.float32)
        def swap_face(self, frame, face, emb, params): print(f"MockFaceSwapperP.swap_face called."); return frame.copy(), None
    class MockFaceEditorP(MockAIProcessor):
        def edit_face_pose_expression(self, frame, face, params): print(f"MockFaceEditorP.edit_face_pose called."); return frame.copy(), None
        def apply_makeup(self, frame, face, params): print(f"MockFaceEditorP.apply_makeup called."); return frame.copy(), None
    class MockFaceRestorerP(MockAIProcessor):
        def restore_face(self, frame, face, params): print(f"MockFaceRestorerP.restore_face called."); return frame.copy(), None
    class MockFrameEnhancerP(MockAIProcessor):
        def enhance_frame(self, frame, params): print(f"MockFrameEnhancerP.enhance_frame called."); return frame.copy(), None
    class MockFaceMaskP(MockAIProcessor):
        def generate_mask(self, frame, face, params): print(f"MockFaceMaskP.generate_mask called."); return np.zeros(frame.shape[:2], dtype=np.uint8), None


    # Instantiate ProcessingOrchestrator (real or mock based on imports)
    if REAL_COMPONENTS_AVAILABLE: # Use real orchestrator with mock processors
        from src.processing_orchestrator import ProcessingOrchestrator # Re-import to ensure it's the real one
        orchestrator = ProcessingOrchestrator(
            ai_model_mgr_orch,
            MockFaceDetectorP("detector", ai_model_mgr_orch),
            MockFaceLandmarkerP("landmarker", ai_model_mgr_orch),
            MockFaceSwapperP("swapper", ai_model_mgr_orch),
            MockFaceEditorP("editor", ai_model_mgr_orch),
            MockFaceRestorerP("restorer", ai_model_mgr_orch),
            MockFrameEnhancerP("enhancer", ai_model_mgr_orch),
            MockFaceMaskP("mask_gen", ai_model_mgr_orch)
        )
    else: # Use mock orchestrator
        orchestrator = ProcessingOrchestrator(ai_model_mgr_orch) # Mock takes only one arg for this example

    # 4. Instantiate UIManager
    ui_manager = UIManager(settings_mgr, param_mgr, orchestrator, media_mgr_mock, workspace_mgr_mock)

    # 5. Demonstrate UIManager functionality
    print("\n--- Demonstrating UIManager ---")
    ui_manager.apply_language("pt_BR")
    print(f"Translated greeting: {ui_manager.tr('greeting')}")
    print(f"Translated button: {ui_manager.tr('start_button')}")
    ui_manager.apply_theme("dark_mode_custom")

    print("\n--- Simulating UI parameter change ---")
    ui_manager.handle_parameter_widget_changed('detection_threshold_Slider', 0.65) # Matches example_layout_data key

    print("\n--- Simulating backend parameter change notification ---")
    # This assumes param_mgr is the real one or a mock that supports _notify
    if hasattr(param_mgr, '_notify'):
        param_mgr._notify('some_other_param_from_backend', 999)
    else: # If using the basic mock that doesn't have _notify from the start
        print("(Skipping backend notify test as ParameterManager mock is basic)")


    print("\n--- Simulating 'Start Processing' click for an image ---")
    ui_manager.handle_start_processing_click() # This will use media_type="image" by default

    print("\n--- Simulating 'Start Processing' click for a video (short) ---")
    # To test video, we'd need to change the simulated_media_type in handle_start_processing_click
    # For this example, let's modify it directly or add a method to set it.
    # Quick hack for testing:
    ui_manager.media_manager.get_media_path = lambda media_type="video": "dummy_video.mp4" if media_type=="video" else "dummy_image.png"
    
    # Update handle_start_processing_click to pick video for this test run
    original_handle_start = ui_manager.handle_start_processing_click
    def modified_handle_start_for_video_test():
        print("UIManager: Modified handle_start_processing_click for video test.")
        current_processing_params = ui_manager.parameter_manager.get_all_current_values()
        media_path = ui_manager.media_manager.get_media_path("video") # Force video
        source_embeddings = ui_manager.workspace_manager.get_source_embeddings()
        orchestrator_params = { "total_frames_sim": 2, "frame_rate_sim": 1.0, "video_resolution_sim": (50,50)} # Minimal video
        ui_manager.processing_orchestrator.start_video_processing(media_path, orchestrator_params, source_embeddings)
    
    ui_manager.handle_start_processing_click = modified_handle_start_for_video_test
    ui_manager.handle_start_processing_click()
    
    # Wait for simulated video processing to finish (2 frames / 1 FPS = 2 seconds)
    time.sleep(3) 
    # If orchestrator is real, stop it explicitly. Mock orchestrator's start_video_processing is non-blocking.
    if isinstance(orchestrator, ProcessingOrchestrator) and hasattr(orchestrator, 'stop_processing'):
         orchestrator.stop_processing()

    ui_manager.handle_start_processing_click = original_handle_start # Restore original

    print("\n--- UIManager Example Usage Finished ---")
