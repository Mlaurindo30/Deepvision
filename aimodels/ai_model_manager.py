import os
import logging

# Attempt to import cachetools
try:
    from cachetools import LRUCache
except ImportError:
    LRUCache = None # Will be checked in __init__

# Attempt to import the actual SettingsManager
# This path might need to be adjusted based on the actual project structure.
try:
    from app.settings_manager import SettingsManager
except ImportError:
    logging.warning("Actual SettingsManager not found at app.settings_manager. Using PlaceholderSettingsManager.")
    class PlaceholderSettingsManager:
        def __init__(self):
            self._settings = {}
            logging.info("PlaceholderSettingsManager initialized.")
        def get(self, key, default=None):
            value = self._settings.get(key, default)
            logging.debug(f"PlaceholderSettingsManager: Getting {key}, returning {value}")
            return value
        def set(self, key, value):
            logging.debug(f"PlaceholderSettingsManager: Setting {key} to {value}")
            self._settings[key] = value
        def save(self):
            logging.debug(f"PlaceholderSettingsManager: Save called. Current settings: {self._settings}")
    SettingsManager = PlaceholderSettingsManager

# Attempt to import providers
try:
    from .providers import ONNXModelProvider
except ImportError as e:
    # This allows AIModelManager to be imported, but _initialize_providers will fail for ONNX
    logging.error(f"Could not import ONNXModelProvider: {e}. AIModelManager may not function correctly for ONNX models.")
    ONNXModelProvider = None # Will be checked in _initialize_providers

logger = logging.getLogger(__name__)

class AIModelManager:
    """
    Manages AI models, including loading, caching, and providing them.
    """
    def __init__(self, settings_manager_instance=None):
        if LRUCache is None:
            logger.critical("cachetools library is not installed. AIModelManager cannot operate without it. Please install it via 'pip install cachetools'.")
            raise ImportError("cachetools library is not installed. Please install it via 'pip install cachetools'.")

        if settings_manager_instance:
            self.settings_manager = settings_manager_instance
            logger.info("Using provided SettingsManager instance.")
        else:
            self.settings_manager = SettingsManager() # Instantiates actual or placeholder
            logger.info("Initialized new SettingsManager instance.")

        # Define defaults and keys
        default_cache_size = 10
        key_cache_size = 'aimodels.cache.max_size'
        default_model_dir = 'models'
        key_model_dir = 'aimodels.model_directory'

        # Register defaults in SettingsManager if not already present
        if self.settings_manager.get(key_cache_size) is None:
            self.settings_manager.set(key_cache_size, default_cache_size)
            logger.info(f"Setting '{key_cache_size}' not found, registered default: {default_cache_size}")
        
        if self.settings_manager.get(key_model_dir) is None:
            self.settings_manager.set(key_model_dir, default_model_dir)
            logger.info(f"Setting '{key_model_dir}' not found, registered default: {default_model_dir}")

        # Now, read the values (which will be the defaults if just set, or existing values)
        cache_max_size = self.settings_manager.get(key_cache_size, default=default_cache_size)
        if not isinstance(cache_max_size, int) or cache_max_size <= 0:
            logger.warning(f"Invalid cache_max_size '{cache_max_size}' from settings. Overriding with default: {default_cache_size}.")
            cache_max_size = default_cache_size
            # Force-set the validated default back if current value was bad and settings manager is not placeholder
            if not isinstance(self.settings_manager, PlaceholderSettingsManager):
                 self.settings_manager.set(key_cache_size, cache_max_size)
                 if hasattr(self.settings_manager, 'save'): self.settings_manager.save()


        self.model_cache = LRUCache(maxsize=cache_max_size)
        self.providers = {} # To store instantiated model providers by name

        # Initialize and register providers
        self._initialize_providers()

        # Get model directory (will use default if not set, or existing value)
        self.model_base_dir = self.settings_manager.get(key_model_dir, default=default_model_dir)
        try:
            if not os.path.isabs(self.model_base_dir):
                # Assuming model_base_dir should be relative to a known root if not absolute.
                # For now, let's log if it's not absolute. Consider project's base path if needed.
                logger.debug(f"Model base directory '{self.model_base_dir}' is a relative path.")
            
            if not os.path.exists(self.model_base_dir):
                os.makedirs(self.model_base_dir, exist_ok=True) 
                logger.info(f"Created model directory: {self.model_base_dir}")
            else:
                logger.info(f"Model directory already exists: {self.model_base_dir}")
        except OSError as e:
            logger.error(f"Error creating model directory {self.model_base_dir}: {e}. Model loading may fail.")
            # Depending on requirements, might re-raise or handle as critical failure
        
        logger.info(f"AIModelManager initialized. Cache size: {cache_max_size}, Model base directory: '{self.model_base_dir}'")

    def _initialize_providers(self):
        # ONNX Provider
        if ONNXModelProvider: # Check if import was successful
            try:
                onnx_provider = ONNXModelProvider()
                self.register_provider("ONNX", onnx_provider)
            except Exception as e: # Catch any error during ONNXModelProvider instantiation
                logger.error(f"Failed to instantiate or register ONNXModelProvider: {e}. ONNX models may not be loadable.")
        else:
            logger.warning("ONNXModelProvider was not imported. ONNX support will be unavailable.")
        
        # Future providers (TensorRT, DFM) can be initialized here similarly:
        # try:
        #     tensorrt_provider = TensorRTModelProvider() # Assuming it exists
        #     self.register_provider("TensorRT", tensorrt_provider)
        # except ImportError as e:
        #     logger.error(f"Failed to initialize TensorRTModelProvider: {e}.")
        # except Exception as e:
        #     logger.error(f"Failed to instantiate TensorRTModelProvider: {e}.")


    def register_provider(self, name: str, provider_instance):
        """
        Registers a model provider instance.
        Args:
            name: The name to identify the provider (e.g., "ONNX").
            provider_instance: An instance of a class that implements the ModelProvider interface.
        """
        if not hasattr(provider_instance, 'load_model') or not hasattr(provider_instance, 'get_model_info'):
             logger.error(f"Provider '{name}' does not conform to ModelProvider interface. Registration failed.")
             raise ValueError(f"Provider '{name}' does not seem to implement the ModelProvider interface (missing load_model or get_model_info).")
        
        self.providers[name] = provider_instance
        logger.info(f"Model provider '{name}' registered successfully.")

    def get_model(self, model_name: str, provider_name: str = "ONNX", model_sub_path: str = None, **kwargs) -> object:
        """
        Retrieves a model, loading it if necessary and using the cache.

        Args:
            model_name: The logical name of the model (e.g., "face_detector_yolov8n").
                        Used for logging and potentially by the provider.
            provider_name: The name of the provider to use (e.g., "ONNX").
            model_sub_path: Optional. The specific path to the model file or directory,
                            relative to `self.model_base_dir`. If None, `model_name`
                            might be used as the filename (e.g. "yolov8n.onnx").
            **kwargs: Additional settings passed to the provider's load_model method.

        Returns:
            The loaded model object (e.g., an ONNX session), or None if loading fails.

        Raises:
            ValueError: If the provider is not supported or if model_sub_path/model_name is invalid.
            FileNotFoundError: If the model file cannot be found.
            RuntimeError: For general model loading failures.
        """
        if not model_name:
            logger.error("get_model called with empty model_name.")
            raise ValueError("model_name cannot be empty.")

        # effective_model_identifier is used for the cache key and to determine the file path component.
        # If model_sub_path is "face_detection/yolo.onnx", use that.
        # If model_sub_path is None, and model_name is "yolo.onnx", use "yolo.onnx".
        effective_model_identifier = model_sub_path if model_sub_path else model_name
        if not effective_model_identifier: # Should not happen if model_name is required
             logger.error("Could not determine a valid model identifier for path or cache key.")
             raise ValueError("Cannot determine model identifier for path construction.")

        cache_key = f"{provider_name}_{effective_model_identifier}"
        logger.debug(f"Attempting to get model. Cache key: '{cache_key}'")

        if cache_key in self.model_cache:
            logger.info(f"Cache HIT for model: '{cache_key}'")
            return self.model_cache[cache_key]
        else:
            logger.info(f"Cache MISS for model: '{cache_key}'. Attempting to load.")
            
            provider = self.providers.get(provider_name)
            if not provider:
                logger.error(f"No provider registered for '{provider_name}'. Cannot load model '{model_name}'.")
                raise ValueError(f"Unsupported model provider: {provider_name}")

            actual_model_path = os.path.join(self.model_base_dir, effective_model_identifier)
            
            if not os.path.exists(actual_model_path):
                logger.error(f"Model file not found at: '{actual_model_path}' (Base dir: '{self.model_base_dir}', Identifier: '{effective_model_identifier}')")
                raise FileNotFoundError(f"Model file not found: {actual_model_path}")

            try:
                logger.info(f"Loading model '{model_name}' (key: '{cache_key}') from '{actual_model_path}' using provider '{provider_name}'.")
                
                # Pass the logical model_name and the full actual_model_path to the provider
                loaded_model_session = provider.load_model(model_path=actual_model_path, model_name=model_name, settings=kwargs)
                
                if loaded_model_session:
                    self.model_cache[cache_key] = loaded_model_session
                    logger.info(f"Model '{cache_key}' loaded and cached successfully.")
                    return loaded_model_session
                else:
                    # This case should ideally not be reached if providers raise exceptions on failure.
                    logger.error(f"Provider '{provider_name}' returned None for model '{model_name}' at path '{actual_model_path}' without raising an exception.")
                    # Depending on strictness, could raise RuntimeError here.
                    return None 
            except FileNotFoundError: # Provider might do its own checks and raise this.
                logger.error(f"File not found during load attempt for model '{model_name}' (key: '{cache_key}') at '{actual_model_path}'. This might be a redundant log if provider also logs.")
                raise # Re-raise to ensure caller knows.
            except Exception as e:
                # Catching general exceptions from provider.load_model (e.g., ONNX Runtime errors)
                logger.error(f"Error loading model '{model_name}' (key: '{cache_key}') using provider '{provider_name}': {e}")
                raise RuntimeError(f"Failed to load model '{model_name}' (Path: {actual_model_path}): {e}") from e

    def get_model_info(self, loaded_model_session: object, provider_name: str = "ONNX") -> dict:
        """
        Retrieves information or metadata about the loaded model session using its provider.

        Args:
            loaded_model_session: The model object/session returned by get_model.
            provider_name: The name of the provider that loaded this model.

        Returns:
            A dictionary containing model information.

        Raises:
            ValueError: If the provider is not supported or if loaded_model_session is invalid.
        """
        logger.debug(f"Attempting to get model info for a session using provider '{provider_name}'.")
        provider = self.providers.get(provider_name)
        if not provider:
            logger.error(f"No provider registered for '{provider_name}'. Cannot get model info.")
            raise ValueError(f"Unsupported model provider: {provider_name}")

        if not hasattr(provider, 'get_model_info'):
            logger.error(f"Provider '{provider_name}' does not have a get_model_info method.")
            raise AttributeError(f"Provider '{provider_name}' does not implement get_model_info.")
            
        try:
            return provider.get_model_info(loaded_model_session)
        except Exception as e:
            logger.error(f"Error retrieving model info using provider '{provider_name}': {e}")
            raise RuntimeError(f"Failed to get model info: {e}") from e

    def unload_model(self, model_name: str = None, provider_name: str = "ONNX", model_sub_path: str = None, cache_key: str = None):
        """
        Removes a model from the cache. If the model provider has an unload_model method,
        it will be called.

        Args:
            model_name: Logical name of the model.
            provider_name: Name of the provider.
            model_sub_path: Sub-path of the model.
            cache_key: Directly provide the cache key to unload. If None, it's constructed
                       from provider_name, model_sub_path, and model_name.
        """
        if not cache_key:
            if not model_name and not model_sub_path: # Need at least one to form a key
                logger.error("Cannot unload model: model_name/model_sub_path or cache_key must be provided.")
                raise ValueError("Either model_name/model_sub_path or a direct cache_key must be provided to unload.")
            effective_model_identifier = model_sub_path if model_sub_path else model_name
            cache_key = f"{provider_name}_{effective_model_identifier}"

        logger.info(f"Attempting to unload model with cache key: '{cache_key}'.")
        if cache_key in self.model_cache:
            model_session = self.model_cache.pop(cache_key) # Remove from cache and get session
            logger.info(f"Model '{cache_key}' removed from cache.")

            # Check if the provider has a specific unload method
            provider = self.providers.get(provider_name)
            if provider and hasattr(provider, 'unload_model'):
                try:
                    logger.debug(f"Calling unload_model on provider '{provider_name}' for model '{cache_key}'.")
                    provider.unload_model(model_session)
                    logger.info(f"Provider '{provider_name}' successfully processed unload for '{cache_key}'.")
                except Exception as e:
                    logger.error(f"Error during provider's unload_model for '{cache_key}': {e}")
            # If no specific unload_model, Python's garbage collector will handle the session object
            # once all references are gone. For ONNXRuntime, this is typically sufficient.
        else:
            logger.warning(f"Model with cache key '{cache_key}' not found in cache. Cannot unload.")
            
    def clear_cache(self):
        """
        Clears the entire model cache.
        For models that need explicit cleanup, this iterates through them.
        """
        logger.info(f"Clearing all models from cache. Current cache size: {len(self.model_cache)}")
        # Iterate over a copy of keys if providers might modify cache during unload
        keys_to_unload = list(self.model_cache.keys())
        
        for cache_key in keys_to_unload:
            # Infer provider_name from cache_key (assuming format "PROVIDER_MODELID")
            try:
                provider_name_in_key = cache_key.split('_', 1)[0]
                # We are calling self.unload_model which will pop from cache
                # and call provider's unload if available.
                self.unload_model(cache_key=cache_key, provider_name=provider_name_in_key)
            except Exception as e: # Catch errors from unload_model if any
                logger.error(f"Error while trying to unload model with key {cache_key} during clear_cache: {e}")

        # self.model_cache.clear() # unload_model already pops, so this might be redundant or an error
        # LRUCache does not have a .clear() that also calls __delitem__ or pops.
        # The loop above is more robust for providers with custom unload logic.
        # If no providers have custom unload, `self.model_cache.clear()` would be simpler,
        # but then provider unload methods wouldn't be called.
        if not self.model_cache: # Check if cache is empty after loop
             logger.info("Model cache successfully cleared.")
        else:
             logger.warning(f"Model cache may not be fully cleared. Remaining items: {len(self.model_cache)}")


# Example Usage (for testing purposes, not part of the module's primary code)
if __name__ == '__main__':
    # Configure basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a dummy ONNX model file for testing
    MODELS_DIR = "temp_models_for_testing"
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # Dummy model content (not a valid ONNX, but good enough for path testing)
    # In a real test, you'd use a minimal valid ONNX file.
    dummy_model_name = "dummy_model.onnx"
    dummy_model_path = os.path.join(MODELS_DIR, dummy_model_name)
    
    # Create a placeholder ONNXModelProvider that doesn't actually load onnxruntime
    # if ONNXModelProvider is None, to allow this test to run without onnxruntime.
    if ONNXModelProvider is None:
        logger.warning("Using dummy ONNXModelProvider for __main__ test as real one failed to import.")
        class DummyONNXModelProvider:
            def load_model(self, model_path: str, model_name: str, settings: dict = None):
                logger.info(f"[DummyProvider] load_model called for {model_name} at {model_path}")
                if not os.path.exists(model_path): raise FileNotFoundError(model_path)
                return f"loaded_session_for_{model_name}" # Dummy session object
            def get_model_info(self, loaded_model: object) -> dict:
                logger.info(f"[DummyProvider] get_model_info called for {loaded_model}")
                return {"name": str(loaded_model), "type": "DummyONNX"}
            def unload_model(self, loaded_model: object):
                logger.info(f"[DummyProvider] unload_model called for {loaded_model}")
        ONNXModelProviderToUse = DummyONNXModelProvider
    else:
        ONNXModelProviderToUse = ONNXModelProvider


    # Test Setup
    try:
        with open(dummy_model_path, "w") as f:
            f.write("This is not a real ONNX model.")
        
        # Test AIModelManager
        # Use placeholder settings that define the model directory
        class TestSettingsManager(PlaceholderSettingsManager): # Inherit from placeholder to get set/get
            def __init__(self):
                super().__init__()
                self._settings['aimodels.model_directory'] = MODELS_DIR
                self._settings['aimodels.cache.max_size'] = 3 # Small cache for testing eviction
        
        settings_mgr = TestSettingsManager()
        manager = AIModelManager(settings_manager_instance=settings_mgr)

        # Manually register the (potentially dummy) ONNX provider if auto-init failed due to no ONNXModelProvider
        if "ONNX" not in manager.providers and ONNXModelProviderToUse:
             logger.info("Manually registering ONNXModelProviderToUse for test.")
             manager.register_provider("ONNX", ONNXModelProviderToUse())


        if "ONNX" in manager.providers:
            # Test 1: Load a model
            logger.info("\n--- Test 1: Load model ---")
            model_session = manager.get_model(model_name=dummy_model_name, provider_name="ONNX")
            assert model_session is not None
            logger.info(f"Loaded model session: {model_session}")

            # Test 2: Cache hit
            logger.info("\n--- Test 2: Cache hit ---")
            model_session_cached = manager.get_model(model_name=dummy_model_name, provider_name="ONNX")
            assert model_session_cached == model_session # Should be the same object from cache
            
            # Test 3: Get model info
            logger.info("\n--- Test 3: Get model info ---")
            if model_session:
                info = manager.get_model_info(model_session, provider_name="ONNX")
                logger.info(f"Model info: {info}")
                assert info is not None

            # Test 4: Unload model
            logger.info("\n--- Test 4: Unload model ---")
            manager.unload_model(model_name=dummy_model_name, provider_name="ONNX")
            assert f"ONNX_{dummy_model_name}" not in manager.model_cache
            logger.info(f"Cache contents after unload: {manager.model_cache}")
            
            # Test 5: Load again after unload (cache miss)
            logger.info("\n--- Test 5: Load again after unload ---")
            model_session_reloaded = manager.get_model(model_name=dummy_model_name, provider_name="ONNX")
            assert model_session_reloaded is not None
            assert model_session_reloaded != model_session # Should be a new session object (if provider creates new ones)
                                                        # or same if provider returns same object for same path.
                                                        # For dummy, it will be new.

            # Test 6: Cache eviction (if cachetools and small cache size)
            if LRUCache is not None and settings_mgr.get('aimodels.cache.max_size', 0) > 0:
                logger.info("\n--- Test 6: Cache eviction ---")
                # Load more models to potentially evict the first one
                for i in range(settings_mgr.get('aimodels.cache.max_size')):
                    name = f"evict_model_{i}.onnx"
                    path = os.path.join(MODELS_DIR, name)
                    with open(path, "w") as f: f.write("dummy")
                    manager.get_model(model_name=name, provider_name="ONNX")
                
                # Check if the first model (dummy_model_name or model_session_reloaded) is still in cache
                # This depends on LRU behavior and exact sequence.
                # For this test, just ensure cache size doesn't exceed max_size
                logger.info(f"Cache size after loading multiple models: {len(manager.model_cache)}")
                assert len(manager.model_cache) <= settings_mgr.get('aimodels.cache.max_size')
            
            # Test 7: Clear cache
            logger.info("\n--- Test 7: Clear cache ---")
            manager.clear_cache()
            assert len(manager.model_cache) == 0
            logger.info("Cache cleared.")

        else:
            logger.error("ONNX provider not registered, skipping functional tests.")

    except Exception as e:
        logger.exception(f"Error during AIModelManager __main__ test: {e}")
    finally:
        # Clean up dummy model and directory
        if os.path.exists(dummy_model_path):
            os.remove(dummy_model_path)
        # remove other dummy models created for eviction test
        for i in range(settings_mgr.get('aimodels.cache.max_size', 0) if 'settings_mgr' in locals() else 0):
            name = f"evict_model_{i}.onnx"
            path = os.path.join(MODELS_DIR, name)
            if os.path.exists(path): os.remove(path)
        if os.path.exists(MODELS_DIR):
            try:
                os.rmdir(MODELS_DIR) # Only if empty
            except OSError:
                logger.warning(f"Could not remove temp models directory {MODELS_DIR} as it might not be empty.")
        logger.info("Test cleanup finished.")
