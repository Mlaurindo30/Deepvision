from abc import ABC, abstractmethod

class ModelProvider(ABC):
    """
    Abstract base class for a model provider.

    This interface defines the contract for loading and managing
    AI models from different sources or runtimes (e.g., ONNX, TensorRT).
    """

    @abstractmethod
    def load_model(self, model_path: str, model_name: str, settings: dict = None) -> object:
        """
        Loads a model from the given path.

        Args:
            model_path: The full path to the model file or directory.
            model_name: The identifier or name of the model.
            settings: Optional dictionary of settings specific to this model or provider.

        Returns:
            An object representing the loaded model (e.g., an ONNX session).

        Raises:
            FileNotFoundError: If the model_path does not exist.
            Exception: For other loading errors.
        """
        pass

    @abstractmethod
    def get_model_info(self, loaded_model: object) -> dict:
        """
        Retrieves information or metadata about the loaded model.

        Args:
            loaded_model: The model object returned by load_model.

        Returns:
            A dictionary containing model information (e.g., input/output shapes, version).
        """
        pass

    # Potentially add other common methods like:
    # @abstractmethod
    # def unload_model(self, loaded_model: object) -> None:
    #     pass
    #
    # @abstractmethod
    # def check_health(self) -> bool:
    #     pass
