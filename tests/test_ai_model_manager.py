import pytest
import os
import shutil
from unittest.mock import patch, MagicMock, call

# Assuming the 'aimodels' module is in the parent directory or installed
# Adjust the import path as necessary for your project structure.
# For example, if 'aimodels' is a top-level directory and tests are in 'tests/':
# from ..aimodels.ai_model_manager import AIModelManager
# from ..aimodels.providers.onnx_provider import ONNXModelProvider
# from ..aimodels.providers.base_provider import ModelProvider

# If your structure is app/aimodels, and app is in PYTHONPATH:
from aimodels.ai_model_manager import AIModelManager
from aimodels.providers.onnx_provider import ONNXModelProvider
from aimodels.providers.base_provider import ModelProvider

# Attempt to import the actual SettingsManager, or use a mock if not found/desired for tests
try:
    from app.settings_manager import SettingsManager 
except ImportError:
    # Using a MagicMock as a placeholder if the real SettingsManager is not available
    # or if we want to ensure tests don't depend on it.
    class PlaceholderSettingsManager:
        def __init__(self):
            self._settings = {}
        def get(self, key, default=None):
            return self._settings.get(key, default)
        def set(self, key, value):
            self._settings[key] = value
        def save(self):
            pass # No-op for placeholder
    SettingsManager = PlaceholderSettingsManager


# --- Constants for Test Data ---
# These paths are relative to the location of this test file (tests/test_ai_model_manager.py)
# The dummy models were created in tests/test_data/models/onnx/
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
ONNX_MODELS_DIR = os.path.join(TEST_DATA_DIR, "models", "onnx")

DUMMY_MODEL_1_NAME = "dummy_model_1.onnx"
DUMMY_MODEL_1_PATH = os.path.join(ONNX_MODELS_DIR, DUMMY_MODEL_1_NAME)

DUMMY_MODEL_2_NAME = "dummy_model_2.onnx"
DUMMY_MODEL_2_PATH = os.path.join(ONNX_MODELS_DIR, DUMMY_MODEL_2_NAME)

INVALID_MODEL_NAME = "invalid_model.txt"
INVALID_MODEL_PATH = os.path.join(ONNX_MODELS_DIR, INVALID_MODEL_NAME)

NON_EXISTENT_MODEL_NAME = "non_existent_model.onnx"


# --- Fixtures ---

@pytest.fixture
def mock_settings_manager_instance():
    """ Provides a MagicMock instance of a SettingsManager. """
    sm_instance = MagicMock(spec=SettingsManager) # Use spec for more accurate mocking
    
    # Default settings for most tests
    default_settings = {
        'aimodels.cache.max_size': 5,
        'aimodels.model_directory': ONNX_MODELS_DIR # Point to actual test data
    }
    
    def get_side_effect(key, default=None):
        return default_settings.get(key, default)
        
    sm_instance.get.side_effect = get_side_effect
    sm_instance.set = MagicMock() # Mock the set method
    sm_instance.save = MagicMock() # Mock the save method
    return sm_instance

@pytest.fixture
def onnx_provider():
    """ Returns an instance of ONNXModelProvider. """
    return ONNXModelProvider()

@pytest.fixture
def ai_model_manager_instance(mock_settings_manager_instance):
    """ 
    Returns an AIModelManager instance initialized with a mocked SettingsManager.
    The model_base_dir will point to the test_data ONNX models.
    """
    # Ensure ONNXModelProvider is found for AIModelManager's _initialize_providers
    # This relies on the default import working. If not, this fixture might need patching.
    manager = AIModelManager(settings_manager_instance=mock_settings_manager_instance)
    return manager

@pytest.fixture
def ai_model_manager_small_cache(tmp_path):
    """
    AIModelManager with a small cache (size 2) and a temporary model directory.
    This fixture is more self-contained for caching tests.
    """
    # Specific SettingsManager for small cache tests
    sm_small_cache = MagicMock(spec=SettingsManager)
    temp_model_dir = tmp_path / "temp_models"
    temp_model_dir.mkdir()

    settings_values = {
        'aimodels.cache.max_size': 2,
        'aimodels.model_directory': str(temp_model_dir)
    }
    sm_small_cache.get.side_effect = lambda key, default=None: settings_values.get(key, default)
    sm_small_cache.set = MagicMock()
    sm_small_cache.save = MagicMock()

    # Create a few dummy model files in the temporary directory for the manager to "load"
    # These don't need to be valid ONNX for caching tests if ONNXModelProvider.load_model is mocked
    for i in range(1, 5):
        with open(temp_model_dir / f"model{i}.onnx", "w") as f:
            f.write(f"dummy onnx content for model{i}")

    manager = AIModelManager(settings_manager_instance=sm_small_cache)
    return manager


# --- Tests for ONNXModelProvider ---

def test_onnx_provider_load_success(onnx_provider):
    """ Test loading a valid ONNX model. """
    assert os.path.exists(DUMMY_MODEL_1_PATH), f"Test model not found: {DUMMY_MODEL_1_PATH}"
    session = onnx_provider.load_model(DUMMY_MODEL_1_PATH, "dummy_model_1")
    assert session is not None
    # Basic check: session should have get_inputs method (part of onnxruntime.InferenceSession)
    assert hasattr(session, 'get_inputs')

def test_onnx_provider_load_file_not_found(onnx_provider):
    """ Test FileNotFoundError when model path is incorrect. """
    with pytest.raises(FileNotFoundError):
        onnx_provider.load_model(os.path.join(ONNX_MODELS_DIR, NON_EXISTENT_MODEL_NAME), "non_existent")

def test_onnx_provider_load_invalid_model(onnx_provider):
    """ Test loading an invalid/corrupted ONNX model. """
    assert os.path.exists(INVALID_MODEL_PATH), f"Invalid test model not found: {INVALID_MODEL_PATH}"
    with pytest.raises(RuntimeError): # onnxruntime typically raises RuntimeError for invalid models
        onnx_provider.load_model(INVALID_MODEL_PATH, "invalid_model")

def test_onnx_provider_get_model_info(onnx_provider):
    """ Test get_model_info for a loaded model. """
    session = onnx_provider.load_model(DUMMY_MODEL_1_PATH, "dummy_model_1")
    info = onnx_provider.get_model_info(session)
    
    assert info["model_type"] == "ONNX"
    assert info["model_path"] == DUMMY_MODEL_1_PATH
    assert "input1" in info["input_names"] # From dummy_model_1 creation
    assert "output1" in info["output_names"] # From dummy_model_1 creation
    assert isinstance(info["input_shapes"], list)
    assert isinstance(info["output_shapes"], list)


# --- Tests for AIModelManager Caching (LRU behavior) ---

@patch('aimodels.providers.onnx_provider.ONNXModelProvider.load_model')
def test_aimanager_cache_hit(mock_load_model, ai_model_manager_small_cache):
    """ Test cache hit: load_model on provider should only be called once for the same model. """
    # Mock the provider's load_model to return a simple string indicating it was called
    mock_load_model.side_effect = lambda model_path, model_name, settings: f"loaded_{model_name}_from_{model_path}"
    
    manager = ai_model_manager_small_cache # Uses cache size 2, temp model dir

    # First load - should call provider's load_model
    model_session1_first_load = manager.get_model(model_name="model1.onnx", provider_name="ONNX")
    assert model_session1_first_load == f"loaded_model1.onnx_from_{manager.model_base_dir}/model1.onnx"
    mock_load_model.assert_called_once_with(model_path=os.path.join(manager.model_base_dir, "model1.onnx"), model_name="model1.onnx", settings={})

    # Second load of the same model - should be a cache hit
    model_session1_second_load = manager.get_model(model_name="model1.onnx", provider_name="ONNX")
    assert model_session1_second_load == model_session1_first_load
    # Provider's load_model should still only have been called once
    mock_load_model.assert_called_once() # No new calls

@patch('aimodels.providers.onnx_provider.ONNXModelProvider.load_model')
def test_aimanager_cache_miss(mock_load_model, ai_model_manager_small_cache):
    """ Test cache miss: loading a new model should call provider's load_model. """
    mock_load_model.side_effect = lambda model_path, model_name, settings: f"loaded_{model_name}_from_{model_path}"
    manager = ai_model_manager_small_cache

    manager.get_model(model_name="model1.onnx", provider_name="ONNX") # Load first model
    expected_path1 = os.path.join(manager.model_base_dir, "model1.onnx")
    mock_load_model.assert_called_once_with(model_path=expected_path1, model_name="model1.onnx", settings={})
    
    manager.get_model(model_name="model2.onnx", provider_name="ONNX") # Load second, different model
    expected_path2 = os.path.join(manager.model_base_dir, "model2.onnx")
    # Check that load_model was called for the second model
    assert mock_load_model.call_count == 2
    mock_load_model.assert_called_with(model_path=expected_path2, model_name="model2.onnx", settings={}) # Check last call

@patch('aimodels.providers.onnx_provider.ONNXModelProvider.load_model')
def test_aimanager_cache_lru_eviction(mock_load_model, ai_model_manager_small_cache):
    """ Test cache eviction: fill cache, load more, check LRU model is evicted. """
    # Cache size is 2 for ai_model_manager_small_cache
    mock_load_model.side_effect = lambda model_path, model_name, settings: f"loaded_{model_name}_from_{model_path}"
    manager = ai_model_manager_small_cache

    model_base = manager.model_base_dir

    # Load model1.onnx (cache: [m1])
    manager.get_model(model_name="model1.onnx", provider_name="ONNX")
    mock_load_model.assert_called_once_with(model_path=os.path.join(model_base, "model1.onnx"), model_name="model1.onnx", settings={})
    
    # Load model2.onnx (cache: [m2, m1])
    manager.get_model(model_name="model2.onnx", provider_name="ONNX")
    assert mock_load_model.call_count == 2
    mock_load_model.assert_called_with(model_path=os.path.join(model_base, "model2.onnx"), model_name="model2.onnx", settings={})

    # Load model3.onnx (cache: [m3, m2], m1 should be evicted)
    manager.get_model(model_name="model3.onnx", provider_name="ONNX")
    assert mock_load_model.call_count == 3
    mock_load_model.assert_called_with(model_path=os.path.join(model_base, "model3.onnx"), model_name="model3.onnx", settings={})
    
    # Verify model1.onnx is no longer in cache (by trying to load it again, should trigger provider.load_model)
    manager.get_model(model_name="model1.onnx", provider_name="ONNX")
    assert mock_load_model.call_count == 4 # Called again for model1
    mock_load_model.assert_called_with(model_path=os.path.join(model_base, "model1.onnx"), model_name="model1.onnx", settings={})

    # Load model2.onnx again (should be a cache hit as it was more recently used than model3 initially, before m1 was reloaded)
    # Cache state before this call (after m1 reloaded): [m1, m3]
    manager.get_model(model_name="model3.onnx", provider_name="ONNX") # Access m3 to make it MRU
    assert mock_load_model.call_count == 4 # No new call, m3 should be in cache

    manager.get_model(model_name="model2.onnx", provider_name="ONNX") # m2 was evicted, should load
    assert mock_load_model.call_count == 5 # Called again for model2


# --- Tests for AIModelManager.get_model (Full Integration) ---

def test_aimanager_get_model_success(ai_model_manager_instance):
    """ Test successful model retrieval using actual ONNX provider (cache miss then hit). """
    manager = ai_model_manager_instance # Uses ONNX_MODELS_DIR
    
    # First load (cache miss)
    session1 = manager.get_model(model_name=DUMMY_MODEL_1_NAME, provider_name="ONNX")
    assert session1 is not None
    assert hasattr(session1, 'get_inputs') # Is an ONNX session

    # Second load (cache hit)
    session2 = manager.get_model(model_name=DUMMY_MODEL_1_NAME, provider_name="ONNX")
    assert session2 is session1 # Should be the exact same object from cache

def test_aimanager_get_model_file_not_found(ai_model_manager_instance):
    """ Test get_model when the model file doesn't exist. """
    manager = ai_model_manager_instance
    with pytest.raises(FileNotFoundError):
        manager.get_model(model_name=NON_EXISTENT_MODEL_NAME, provider_name="ONNX")

def test_aimanager_get_model_unsupported_provider(ai_model_manager_instance):
    """ Test get_model with an unsupported provider name. """
    manager = ai_model_manager_instance
    with pytest.raises(ValueError, match="Unsupported model provider: FAKE_PROVIDER"):
        manager.get_model(model_name=DUMMY_MODEL_1_NAME, provider_name="FAKE_PROVIDER")

def test_aimanager_get_model_path_construction(mock_settings_manager_instance, tmp_path):
    """ Test that get_model correctly constructs paths using model_directory from settings. """
    
    # Create a temporary, uniquely named model directory for this test
    specific_test_model_dir = tmp_path / "custom_model_dir_for_path_test"
    specific_test_model_dir.mkdir()
    
    # Put a dummy model in this custom directory
    dummy_model_in_custom_dir = specific_test_model_dir / "model_in_custom.onnx"
    with open(dummy_model_in_custom_dir, "w") as f:
        f.write("dummy onnx for path construction test") # Content doesn't need to be valid if load_model is mocked

    # Update the mocked SettingsManager for this specific test path
    mock_settings_manager_instance.get.side_effect = lambda key, default=None: {
        'aimodels.cache.max_size': 5,
        'aimodels.model_directory': str(specific_test_model_dir) # Point to custom dir
    }.get(key, default)

    # Patch the ONNXModelProvider's load_model for this test to avoid real loading
    # and to inspect the path it was called with.
    with patch.object(ONNXModelProvider, 'load_model', return_value="mock_session") as mock_load:
        manager = AIModelManager(settings_manager_instance=mock_settings_manager_instance)
        
        # Test with model_name (e.g., "model_in_custom.onnx")
        manager.get_model(model_name="model_in_custom.onnx", provider_name="ONNX")
        expected_path1 = os.path.join(str(specific_test_model_dir), "model_in_custom.onnx")
        mock_load.assert_called_with(model_path=expected_path1, model_name="model_in_custom.onnx", settings={})

        # Test with model_sub_path
        sub_path_model_name = "sub_path_model.onnx"
        sub_path_full = specific_test_model_dir / "category" / sub_path_model_name
        (specific_test_model_dir / "category").mkdir()
        with open(sub_path_full, "w") as f: f.write("dummy")

        manager.get_model(model_name="logical_name_for_subpath", 
                          provider_name="ONNX", 
                          model_sub_path=os.path.join("category", sub_path_model_name))
        expected_path2 = str(sub_path_full)
        mock_load.assert_called_with(model_path=expected_path2, model_name="logical_name_for_subpath", settings={})


# --- Helper for cleaning up test_data if needed, though typically not required with .gitignore ---
# @pytest.fixture(scope="session", autouse=True)
# def cleanup_test_data(request):
#     """ Session-scoped fixture to clean up created test files if necessary. """
#     def fin():
#         # Example: remove the ONNX_MODELS_DIR if it was created by tests and not part of repo
#         # Be very careful with this if test_data is checked into git.
#         # if os.path.exists(ONNX_MODELS_DIR) and "test_data_generated_by_tests" in ONNX_MODELS_DIR:
#         #     shutil.rmtree(ONNX_MODELS_DIR)
#         pass
#     # request.addfinalizer(fin) # Only add if cleanup is actually needed and safe

if __name__ == "__main__":
    # This allows running pytest directly on this file:
    # python -m pytest tests/test_ai_model_manager.py
    pytest.main([__file__])

# Future tests to consider:
# - AIModelManager: test unload_model, clear_cache
# - AIModelManager: test behavior when SettingsManager is missing some keys initially
# - AIModelManager: test with multiple providers registered
# - Robustness: test more edge cases for paths, names, etc.
# - Concurrency: If relevant, test thread safety of cache (though LRUCache itself is not thread-safe by default)
# - Logging: Verify log messages are emitted (e.g. using caplog fixture)
# - Test ONNXModelProvider's unload_model method if it has specific logic.
# - Test initialization of AIModelManager if settings are invalid (e.g. cache size is not int)
# - Test AIModelManager's model_base_dir creation if it doesn't exist.

# Note on SettingsManager import:
# The test file attempts to import `app.settings_manager.SettingsManager`.
# If this path is incorrect or `SettingsManager` is part of a larger application
# context that's hard to initialize in tests, the `PlaceholderSettingsManager`
# or a more robust mocking strategy (e.g., patching the import within `aimodels.ai_model_manager`)
# would be essential. The current `mock_settings_manager_instance` fixture provides
# a MagicMock, which is a good way to isolate from the real `SettingsManager`.
# If AIModelManager itself imports settings_manager like `from app.settings_manager import SettingsManager`,
# then to mock it for ALL tests, you'd use something like:
# @patch('aimodels.ai_model_manager.SettingsManager', new_callable=MagicMock)
# in a conftest.py or at the top of the test file if that module is loaded before tests.
# However, passing the mocked instance to AIModelManager constructor is often cleaner if available.
# The current `AIModelManager` code allows passing an instance, which is good for testability.
# The `ai_model_manager_instance` fixture uses this by providing `mock_settings_manager_instance`.
# The `PlaceholderSettingsManager` in this test file is a fallback if `app.settings_manager` is not found,
# ensuring the test file itself can be parsed.
# The `mock_settings_manager_instance` fixture should be what `AIModelManager` uses in most tests.
# The test `test_aimanager_get_model_path_construction` shows how to customize this mock per-test.
