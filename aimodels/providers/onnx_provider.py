# Try to import onnxruntime and raise a clear error if not found.
try:
    import onnxruntime
except ImportError:
    # Option 1: Raise an error immediately
    # raise ImportError("ONNXRuntime is not installed. Please install it to use ONNXModelProvider.")
    # Option 2: Set a flag and check in methods (more complex, prefer Option 1 for now or ensure install)
    onnxruntime = None 

from .base_provider import ModelProvider
import os # For FileNotFoundError

class ONNXModelProvider(ModelProvider):
    """
    Model provider for ONNX models using onnxruntime.
    """

    def __init__(self):
        if onnxruntime is None:
            raise ImportError("ONNXRuntime is not installed. Please install it via 'pip install onnxruntime' to use ONNXModelProvider.")
        self._loaded_model_paths = {} # To store model_path for get_model_info

    def load_model(self, model_path: str, model_name: str, settings: dict = None) -> object:
        """
        Loads an ONNX model from the given path using onnxruntime.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            # For now, provider_options can be empty or configured via settings if needed later
            session = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
            # Store model_path for get_model_info. Using object id as key.
            self._loaded_model_paths[id(session)] = model_path
            return session
        except Exception as e:
            # Catching general Exception from onnxruntime load
            # Log this error appropriately in a real scenario
            print(f"Error loading ONNX model {model_name} from {model_path}: {e}")
            raise RuntimeError(f"Failed to load ONNX model {model_name}: {e}")


    def get_model_info(self, loaded_model: object) -> dict:
        """
        Retrieves information about the loaded ONNX model.
        loaded_model is expected to be an onnxruntime.InferenceSession.
        """
        if onnxruntime is None: # Should have been caught in __init__, but as a safeguard
            raise ImportError("ONNXRuntime is not installed.")
        
        if not isinstance(loaded_model, onnxruntime.InferenceSession):
            raise ValueError("loaded_model is not an ONNX InferenceSession.")

        inputs = loaded_model.get_inputs()
        outputs = loaded_model.get_outputs()
        
        model_path = self._loaded_model_paths.get(id(loaded_model), "Unknown - path not stored or model not loaded by this provider instance")

        return {
            "model_type": "ONNX",
            "model_path": model_path,
            "input_names": [inp.name for inp in inputs],
            "input_shapes": [inp.shape for inp in inputs],
            "output_names": [out.name for out in outputs],
            "output_shapes": [out.shape for out in outputs],
            "execution_providers": loaded_model.get_providers()
        }

    def unload_model(self, loaded_model: object):
        """
        Unloads a model. For ONNX, the session object can be deleted.
        Also remove the path from our tracking dictionary.
        """
        if id(loaded_model) in self._loaded_model_paths:
            del self._loaded_model_paths[id(loaded_model)]
        # For onnxruntime, session deletion is handled by Python's garbage collector
        # If specific cleanup is needed (e.g. C++ resources), it would be done here.
        print(f"Model {id(loaded_model)} unloaded (or will be by GC). Path info removed.")
