import onnxruntime
import os
import threading
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional, Union

# --- Configuration Data Structures (Simulated) ---
SIMULATED_MODELS_DATA = {
    "inswapper_128": {
        "path": "./models/swapper/inswapper_128.onnx", # Relative to app root
        "url": "http://example.com/downloads/inswapper_128.onnx", # Placeholder
        "hash": "dummyhash123", # Placeholder
        "type": "swapper",
        "primary": True # Example: This model is considered primary
    },
    "gfpgan_1.4": {
        "path": "./models/enhancer/gfpgan_1.4.onnx",
        "url": "http://example.com/downloads/gfpgan_1.4.onnx",
        "hash": "dummyhash456",
        "type": "enhancer",
        "primary": False
    },
    "realesrgan": {
        "path": "./models/enhancer/realesrgan.onnx",
        "url": "http://example.com/downloads/realesrgan.onnx",
        "hash": "dummyhash789",
        "type": "enhancer",
        "primary": False
    }
    # Add more models as needed for testing
}

class AIModelManager:
    def __init__(
        self,
        models_data: Dict[str, Dict[str, Any]],
        initial_device_str: str = "cpu", # 'cpu' or 'cuda' (or others ONNX supports)
        cache_limit: int = 3 # Max number of models in cache (excluding primary)
    ):
        self.models_data = models_data
        self.cache_limit = cache_limit
        self._lock = threading.RLock()
        self._onnx_sessions_cache: OrderedDict[str, onnxruntime.InferenceSession] = OrderedDict()
        self._primary_models: List[str] = [
            name for name, data in models_data.items() if data.get("primary", False)
        ]
        
        self.current_providers: List[Union[str, Tuple[str, Dict]]] = []
        self._available_providers = onnxruntime.get_available_providers()
        print(f"ONNXRuntime available providers: {self._available_providers}")

        if initial_device_str == "cuda" and "CUDAExecutionProvider" in self._available_providers:
            self.set_execution_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            if initial_device_str == "cuda":
                print("Warning: CUDAExecutionProvider requested but not available. Falling back to CPU.")
            self.set_execution_providers(["CPUExecutionProvider"])

    def _is_provider_config_valid(self, providers_config_list: List[Union[str, Tuple[str, Dict]]]) -> bool:
        if not providers_config_list: # Must have at least one provider
            print("Error: Execution providers list cannot be empty.")
            return False
        for provider_entry in providers_config_list:
            provider_name = provider_entry if isinstance(provider_entry, str) else provider_entry[0]
            if provider_name not in self._available_providers:
                print(f"Error: Provider '{provider_name}' is not available in this ONNXRuntime build.")
                return False
        return True

    def set_execution_providers(self, providers_config_list: List[Union[str, Tuple[str, Dict]]]) -> bool:
        """
        Sets the execution providers for ONNX Runtime.
        Example: ['CUDAExecutionProvider', 'CPUExecutionProvider'] or [('TensorrtExecutionProvider', {'trt_fp16_enable': True})]
        Clears existing ONNX session cache as providers change.
        """
        with self._lock:
            if not self._is_provider_config_valid(providers_config_list):
                print("Error: Invalid provider configuration. Not applying changes.")
                return False

            print(f"Setting execution providers to: {providers_config_list}")
            self.current_providers = providers_config_list
            self.clear_onnx_cache() # Force reload of models with new providers
            return True

    def _download_model_if_needed(self, model_name: str) -> Optional[str]:
        """
        Placeholder for model downloading logic.
        Checks if the model file exists, if not, tries to download it.
        Returns the path to the model file if successful, None otherwise.
        """
        model_info = self.models_data.get(model_name)
        if not model_info:
            print(f"Error: Model {model_name} not found in models_data.")
            return None

        model_path = model_info["path"]
        
        # Ensure the model path is absolute or relative to a known root (e.g., /app)
        # For this example, assuming paths in SIMULATED_MODELS_DATA are relative to /app
        abs_model_path = os.path.abspath(model_path)

        if not os.path.exists(abs_model_path):
            print(f"Model file {abs_model_path} for {model_name} does not exist. (Download logic placeholder)")
            # Placeholder: In a real scenario, download from model_info["url"]
            # and verify with model_info["hash"]
            # For now, since we created dummy files, this part might not be hit unless a path is wrong.
            # For testing the placeholder, one could temporarily rename a dummy file.
            # Example download:
            # url = model_info.get("url")
            # if url:
            #     print(f"Attempting to download {model_name} from {url}...")
            #     # import requests
            #     # response = requests.get(url)
            #     # response.raise_for_status() # Raise an exception for HTTP errors
            #     # with open(model_path, 'wb') as f:
            #     #     f.write(response.content)
            #     # print(f"Model {model_name} downloaded successfully.")
            # else:
            #     print(f"Error: No URL specified for {model_name} and file not found.")
            #     return None
            return None # If download fails or no URL
        
        return abs_model_path


    def get_onnx_session(self, model_name: str) -> Optional[onnxruntime.InferenceSession]:
        """
        Gets an ONNX InferenceSession. Manages an LRU cache.
        Primary models are less likely to be evicted.
        """
        with self._lock:
            if model_name in self._onnx_sessions_cache:
                # Move accessed item to the end to mark it as recently used (for LRU)
                self._onnx_sessions_cache.move_to_end(model_name)
                print(f"Model {model_name} found in cache.")
                return self._onnx_sessions_cache[model_name]

            model_info = self.models_data.get(model_name)
            if not model_info:
                print(f"Error: Model {model_name} not defined in models_data.")
                return None

            model_path = self._download_model_if_needed(model_name)
            if not model_path:
                print(f"Error: Model file for {model_name} could not be found or downloaded.")
                return None

            try:
                print(f"Loading model {model_name} from {model_path} with providers: {self.current_providers}")
                session_options = onnxruntime.SessionOptions()
                # Further session options can be set here if needed, e.g. graph optimization level
                # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                session = onnxruntime.InferenceSession(
                    model_path,
                    providers=self.current_providers,
                    sess_options=session_options
                )
                
                # Cache management
                if len(self._onnx_sessions_cache) >= self.cache_limit + len(self._primary_models):
                    # Evict LRU if cache is full, prioritizing non-primary models
                    evicted_model = None
                    for cached_name in list(self._onnx_sessions_cache.keys()): # Iterate over a copy
                        if cached_name not in self._primary_models:
                            evicted_model = cached_name
                            break
                    if evicted_model:
                        print(f"Cache limit reached. Evicting {evicted_model}.")
                        del self._onnx_sessions_cache[evicted_model]
                    elif self._primary_models and model_name not in self._primary_models:
                        # This case implies cache is full of primary models, and we are trying to add another non-primary.
                        # This logic might need refinement based on desired behavior (e.g. error, or allow temporary overrun)
                        print(f"Warning: Cache is full of primary models. Consider increasing cache_limit if {model_name} is frequently used.")


                self._onnx_sessions_cache[model_name] = session
                self._onnx_sessions_cache.move_to_end(model_name) # Mark as recently used
                print(f"Model {model_name} loaded and cached successfully.")
                return session

            except Exception as e:
                print(f"Error loading ONNX model {model_name} from {model_path}: {e}")
                return None

    def clear_onnx_cache(self):
        """Clears all cached ONNX sessions."""
        with self._lock:
            print("Clearing ONNX session cache.")
            # Properly release resources if ONNXRuntime sessions have such methods
            # For InferenceSession, Python's garbage collector handles it when references are removed.
            self._onnx_sessions_cache.clear()

    def get_cached_models_count(self) -> int:
        with self._lock:
            return len(self._onnx_sessions_cache)


if __name__ == '__main__':
    print("--- AIModelManager Example Usage ---")

    # Test with default CPU providers first
    print("\n--- Initializing with CPU ---")
    model_manager = AIModelManager(SIMULATED_MODELS_DATA, initial_device_str="cpu", cache_limit=1) # Small cache for testing eviction

    print(f"\nInitial providers: {model_manager.current_providers}")
    print(f"Primary models: {model_manager._primary_models}")

    # 1. Get a primary model
    print("\n--- Requesting primary model 'inswapper_128' ---")
    session1 = model_manager.get_onnx_session("inswapper_128")
    assert session1 is not None, "Failed to load inswapper_128"
    print(f"Cache size: {model_manager.get_cached_models_count()}") # Should be 1

    # 2. Get a non-primary model - should load
    print("\n--- Requesting non-primary model 'gfpgan_1.4' ---")
    session2 = model_manager.get_onnx_session("gfpgan_1.4")
    assert session2 is not None, "Failed to load gfpgan_1.4"
    print(f"Cache size: {model_manager.get_cached_models_count()}") # Should be 2 (1 primary + 1 non-primary in cache_limit)

    # 3. Get another non-primary model - should evict gfpgan_1.4 due to cache_limit=1 for non-primaries
    print("\n--- Requesting non-primary model 'realesrgan' ---")
    session3 = model_manager.get_onnx_session("realesrgan")
    assert session3 is not None, "Failed to load realesrgan"
    print(f"Cache size: {model_manager.get_cached_models_count()}") # Should be 2 (inswapper_128 is primary, realesrgan replaced gfpgan)
    assert "gfpgan_1.4" not in model_manager._onnx_sessions_cache, "gfpgan_1.4 was not evicted!"
    assert "realesrgan" in model_manager._onnx_sessions_cache, "realesrgan was not cached!"


    # 4. Request gfpgan_1.4 again - should load and evict realesrgan
    print("\n--- Requesting 'gfpgan_1.4' again ---")
    session4 = model_manager.get_onnx_session("gfpgan_1.4")
    assert session4 is not None, "Failed to reload gfpgan_1.4"
    print(f"Cache size: {model_manager.get_cached_models_count()}") # Should be 2
    assert "realesrgan" not in model_manager._onnx_sessions_cache, "realesrgan was not evicted!"
    assert "gfpgan_1.4" in model_manager._onnx_sessions_cache, "gfpgan_1.4 was not cached!"

    # 5. Test changing providers (if CUDA is available, otherwise it will stick to CPU)
    print("\n--- Attempting to set CUDAExecutionProvider (if available) ---")
    cuda_available = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    if cuda_available:
        changed_providers = model_manager.set_execution_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert changed_providers, "Failed to set CUDA providers"
        print(f"Providers after change: {model_manager.current_providers}")
        assert model_manager.get_cached_models_count() == 0, "Cache not cleared after provider change"
        
        print("\n--- Requesting 'inswapper_128' with new CUDA providers ---")
        session_cuda = model_manager.get_onnx_session("inswapper_128")
        assert session_cuda is not None, "Failed to load inswapper_128 with CUDA"
        print(f"Session providers: {session_cuda.get_providers()}")
        # Check if CUDA is actually used (the first provider in the list is usually the one chosen if compatible)
        # This check is a bit fragile as it depends on the internal naming ONNXRuntime uses.
        assert "CUDAExecutionProvider" in session_cuda.get_providers()[0], "CUDA provider not primary for session"

    else:
        print("CUDAExecutionProvider not available, skipping CUDA specific tests.")
        changed_providers_fail = model_manager.set_execution_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert not changed_providers_fail, "Setting unavailable CUDA provider should fail or warn and not apply"
        print(f"Providers after attempting to set unavailable CUDA: {model_manager.current_providers}")


    # 6. Test clearing cache
    print("\n--- Clearing cache ---")
    model_manager.clear_onnx_cache()
    assert model_manager.get_cached_models_count() == 0, "Cache not empty after clear"

    # 7. Test model not found
    print("\n--- Requesting non-existent model 'ghost_model' ---")
    session_ghost = model_manager.get_onnx_session("ghost_model")
    assert session_ghost is None, "Ghost model actually returned a session!"

    # 8. Test model file missing (placeholder for download logic)
    # Temporarily rename a model file to test the download path
    original_realesrgan_path = SIMULATED_MODELS_DATA["realesrgan"]["path"]
    temp_realesrgan_path = original_realesrgan_path + ".temp_missing"
    
    if os.path.exists(original_realesrgan_path):
        print(f"\n--- Testing missing model file (temporarily renaming {original_realesrgan_path}) ---")
        os.rename(original_realesrgan_path, temp_realesrgan_path)
        SIMULATED_MODELS_DATA["realesrgan"]["path"] = temp_realesrgan_path # Update manager's view
        # Re-initialize manager or update its models_data if it's meant to be dynamic
        # For this test, creating a new instance to pick up the changed path for "realesrgan"
        # Or, more simply, just try to get it with the current manager, which will now use the bad path.
        model_manager_test_missing = AIModelManager(SIMULATED_MODELS_DATA, initial_device_str="cpu", cache_limit=1)
        session_missing = model_manager_test_missing.get_onnx_session("realesrgan")
        assert session_missing is None, "Session returned for a missing model file (download placeholder not active)"
        # Restore
        os.rename(temp_realesrgan_path, original_realesrgan_path)
        SIMULATED_MODELS_DATA["realesrgan"]["path"] = original_realesrgan_path # Restore for other tests
        print(f"Restored model file {original_realesrgan_path}")
    else:
        print(f"Skipping missing model file test as {original_realesrgan_path} not found initially.")


    # 9. Test invalid provider configuration
    print("\n--- Testing invalid provider configuration ---")
    initial_providers = model_manager.current_providers.copy()
    set_invalid = model_manager.set_execution_providers(["NonExistentProvider", "CPUExecutionProvider"])
    assert not set_invalid, "Setting invalid provider should fail"
    assert model_manager.current_providers == initial_providers, "Providers changed despite invalid config"

    set_empty_invalid = model_manager.set_execution_providers([])
    assert not set_empty_invalid, "Setting empty provider list should fail"
    assert model_manager.current_providers == initial_providers, "Providers changed despite empty invalid config"


    print("\n--- AIModelManager Example Usage Finished ---")
