"""
Manages parameters for different scopes within the application.

This module defines:
- BaseParameter: An abstract base class for parameter types.
- Specific parameter types: IntParameter, FloatParameter, StringParameter, BoolParameter.
- ParameterManager: A class to define, get, and set parameters, managing their
  definitions and current values across different scopes.
"""

from typing import Any, Dict, List, Optional, Union

class BaseParameter:
    """
    Base class for a parameter definition.

    Attributes:
        name (str): The programmatic name of the parameter.
        description (str): A user-friendly description for UI purposes.
        default_value (Any): The default value for this parameter.
        param_type (str): A string representing the type of the parameter.
    """
    param_type = "base"

    def __init__(self, name: str, description: str = "", default_value: Any = None):
        """
        Initializes a BaseParameter.

        Args:
            name (str): The name of the parameter.
            description (str, optional): A description of the parameter. Defaults to "".
            default_value (Any, optional): The default value for the parameter. Defaults to None.
        """
        self.name = name
        self.description = description
        self.default_value = default_value

    def validate(self, value: Any) -> bool:
        """
        Validates the given value against the parameter's constraints.
        The base implementation always returns True.

        Args:
            value (Any): The value to validate.

        Raises:
            ValueError: If the value is invalid (not used in base class).

        Returns:
            bool: True if the value is valid, False otherwise (though typically raises error).
        """
        return True

    def get_details(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the details of the parameter.

        Returns:
            Dict[str, Any]: A dictionary with 'name', 'description', 'type', and 'default_value'.
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.param_type,
            "default_value": self.default_value,
        }

class IntParameter(BaseParameter):
    """
    Parameter definition for integer values.
    """
    param_type = "int"

    def __init__(self, name: str, description: str = "", default_value: int = 0,
                 min_value: Optional[int] = None, max_value: Optional[int] = None):
        """
        Initializes an IntParameter.

        Args:
            name (str): The name of the parameter.
            description (str, optional): A description of the parameter. Defaults to "".
            default_value (int, optional): The default integer value. Defaults to 0.
            min_value (Optional[int], optional): Minimum allowed value. Defaults to None.
            max_value (Optional[int], optional): Maximum allowed value. Defaults to None.
        """
        super().__init__(name, description, default_value)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> bool:
        """
        Validates if the value is an integer and within the defined min/max bounds.

        Args:
            value (Any): The value to validate.

        Raises:
            ValueError: If value is not an int or not within bounds.

        Returns:
            bool: True if valid.
        """
        if not isinstance(value, int):
            raise ValueError(f"Parameter '{self.name}': Value '{value}' must be an integer.")
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Parameter '{self.name}': Value {value} is less than minimum {self.min_value}.")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Parameter '{self.name}': Value {value} is greater than maximum {self.max_value}.")
        return True

    def get_details(self) -> Dict[str, Any]:
        """
        Returns details of the IntParameter, including min/max values if set.
        """
        details = super().get_details()
        if self.min_value is not None:
            details["min_value"] = self.min_value
        if self.max_value is not None:
            details["max_value"] = self.max_value
        return details

class FloatParameter(BaseParameter):
    """
    Parameter definition for float values.
    """
    param_type = "float"

    def __init__(self, name: str, description: str = "", default_value: float = 0.0,
                 min_value: Optional[Union[float, int]] = None,
                 max_value: Optional[Union[float, int]] = None):
        """
        Initializes a FloatParameter.

        Args:
            name (str): The name of the parameter.
            description (str, optional): A description of the parameter. Defaults to "".
            default_value (float, optional): The default float value. Defaults to 0.0.
            min_value (Optional[Union[float, int]], optional): Minimum allowed value. Defaults to None.
            max_value (Optional[Union[float, int]], optional): Maximum allowed value. Defaults to None.
        """
        super().__init__(name, description, float(default_value))
        self.min_value = float(min_value) if min_value is not None else None
        self.max_value = float(max_value) if max_value is not None else None

    def validate(self, value: Any) -> bool:
        """
        Validates if the value is a float (or convertible int) and within min/max bounds.

        Args:
            value (Any): The value to validate.

        Raises:
            ValueError: If value is not a float/int or not within bounds.

        Returns:
            bool: True if valid.
        """
        if not isinstance(value, (float, int)):
            raise ValueError(f"Parameter '{self.name}': Value '{value}' must be a float or an integer.")
        
        float_value = float(value)

        if self.min_value is not None and float_value < self.min_value:
            raise ValueError(f"Parameter '{self.name}': Value {float_value} is less than minimum {self.min_value}.")
        if self.max_value is not None and float_value > self.max_value:
            raise ValueError(f"Parameter '{self.name}': Value {float_value} is greater than maximum {self.max_value}.")
        return True

    def get_details(self) -> Dict[str, Any]:
        """
        Returns details of the FloatParameter, including min/max values if set.
        """
        details = super().get_details()
        if self.min_value is not None:
            details["min_value"] = self.min_value
        if self.max_value is not None:
            details["max_value"] = self.max_value
        return details

class StringParameter(BaseParameter):
    """
    Parameter definition for string values.
    """
    param_type = "string"

    def __init__(self, name: str, description: str = "", default_value: str = "",
                 min_length: Optional[int] = None, max_length: Optional[int] = None,
                 choices: Optional[List[str]] = None):
        """
        Initializes a StringParameter.

        Args:
            name (str): The name of the parameter.
            description (str, optional): A description of the parameter. Defaults to "".
            default_value (str, optional): The default string value. Defaults to "".
            min_length (Optional[int], optional): Minimum string length. Defaults to None.
            max_length (Optional[int], optional): Maximum string length. Defaults to None.
            choices (Optional[List[str]], optional): A list of valid string choices. Defaults to None.
        """
        super().__init__(name, description, default_value)
        self.min_length = min_length
        self.max_length = max_length
        self.choices = choices

    def validate(self, value: Any) -> bool:
        """
        Validates if the value is a string, within length constraints, and in choices (if defined).

        Args:
            value (Any): The value to validate.

        Raises:
            ValueError: If value is not a string or violates constraints.

        Returns:
            bool: True if valid.
        """
        if not isinstance(value, str):
            raise ValueError(f"Parameter '{self.name}': Value '{value}' must be a string.")
        if self.min_length is not None and len(value) < self.min_length:
            raise ValueError(f"Parameter '{self.name}': Length of '{value}' ({len(value)}) is less than minimum {self.min_length}.")
        if self.max_length is not None and len(value) > self.max_length:
            raise ValueError(f"Parameter '{self.name}': Length of '{value}' ({len(value)}) is greater than maximum {self.max_length}.")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"Parameter '{self.name}': Value '{value}' is not in allowed choices: {self.choices}.")
        return True

    def get_details(self) -> Dict[str, Any]:
        """
        Returns details of the StringParameter, including length constraints and choices if set.
        """
        details = super().get_details()
        if self.min_length is not None:
            details["min_length"] = self.min_length
        if self.max_length is not None:
            details["max_length"] = self.max_length
        if self.choices is not None:
            details["choices"] = self.choices
        return details

class BoolParameter(BaseParameter):
    """
    Parameter definition for boolean values.
    """
    param_type = "bool"

    def __init__(self, name: str, description: str = "", default_value: bool = False):
        """
        Initializes a BoolParameter.

        Args:
            name (str): The name of the parameter.
            description (str, optional): A description of the parameter. Defaults to "".
            default_value (bool, optional): The default boolean value. Defaults to False.
        """
        super().__init__(name, description, default_value)

    def validate(self, value: Any) -> bool:
        """
        Validates if the value is a boolean.

        Args:
            value (Any): The value to validate.

        Raises:
            ValueError: If value is not a boolean.

        Returns:
            bool: True if valid.
        """
        if not isinstance(value, bool):
            raise ValueError(f"Parameter '{self.name}': Value '{value}' must be a boolean (True or False).")
        return True

class ParameterManager:
    """
    Manages parameter definitions and their current values across different scopes.
    """
    def __init__(self):
        """
        Initializes the ParameterManager with empty storages for definitions and values.
        """
        self.parameter_definitions: Dict[str, Dict[str, BaseParameter]] = {}
        self.parameter_values: Dict[str, Dict[str, Any]] = {}

    def define_parameters(self, scope: str, param_defs: List[BaseParameter]):
        """
        Registers a list of parameter definitions for a given scope.

        Args:
            scope (str): The scope to which these parameters belong (e.g., "rendering", "data_processing").
            param_defs (List[BaseParameter]): A list of parameter definition instances.
        """
        if scope not in self.parameter_definitions:
            self.parameter_definitions[scope] = {}
            self.parameter_values[scope] = {}

        for param_def in param_defs:
            if not isinstance(param_def, BaseParameter):
                raise TypeError(f"Invalid parameter definition type for '{param_def.name if hasattr(param_def, 'name') else 'unknown'}'. Must be a BaseParameter instance.")
            
            self.parameter_definitions[scope][param_def.name] = param_def
            # Initialize with default value, after validation (though default should always be valid)
            try:
                param_def.validate(param_def.default_value)
                self.parameter_values[scope][param_def.name] = param_def.default_value
            except ValueError as e:
                # This should ideally not happen if default values are correctly set in definitions
                raise ValueError(f"Default value for parameter '{param_def.name}' in scope '{scope}' is invalid: {e}")


    def get_parameter_definitions(self, scope: str) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves the details of all parameter definitions for a specified scope.

        Args:
            scope (str): The scope whose parameter definitions are to be retrieved.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are parameter names and
                                       values are dictionaries of parameter details.
                                       Returns an empty dictionary if the scope does not exist.
        """
        if scope not in self.parameter_definitions:
            return {}
        
        return {
            name: param_def.get_details()
            for name, param_def in self.parameter_definitions[scope].items()
        }

    def get_parameter(self, scope: str, param_name: str) -> Any:
        """
        Retrieves the current value of a specific parameter within a scope.

        Args:
            scope (str): The scope of the parameter.
            param_name (str): The name of the parameter.

        Raises:
            KeyError: If the scope or parameter name does not exist.

        Returns:
            Any: The current value of the parameter.
        """
        if scope not in self.parameter_values:
            raise KeyError(f"Scope '{scope}' not found.")
        if param_name not in self.parameter_values[scope]:
            # This implies it was not defined via define_parameters
            raise KeyError(f"Parameter '{param_name}' not found in scope '{scope}'.")
        return self.parameter_values[scope][param_name]

    def set_parameter(self, scope: str, param_name: str, value: Any):
        """
        Sets the value of a specific parameter within a scope, after validation.

        Args:
            scope (str): The scope of the parameter.
            param_name (str): The name of the parameter to set.
            value (Any): The new value for the parameter.

        Raises:
            KeyError: If the scope or parameter name has not been defined.
            ValueError: If the provided value fails validation for the parameter type.
        """
        if scope not in self.parameter_definitions:
            raise KeyError(f"Scope '{scope}' not found. Define parameters first.")
        if param_name not in self.parameter_definitions[scope]:
            raise KeyError(f"Parameter '{param_name}' not defined in scope '{scope}'. Define parameters first.")

        param_def = self.parameter_definitions[scope][param_name]
        param_def.validate(value)  # This will raise ValueError if invalid

        # If FloatParameter, ensure value is stored as float even if int is passed
        if isinstance(param_def, FloatParameter) and isinstance(value, int):
            self.parameter_values[scope][param_name] = float(value)
        else:
            self.parameter_values[scope][param_name] = value


    def get_all_parameter_values(self, scope: str) -> Dict[str, Any]:
        """
        Retrieves all current parameter names and their values for a specified scope.

        Args:
            scope (str): The scope whose parameters are to be retrieved.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and their current values.
                            Returns an empty dictionary if the scope does not exist.
        """
        return self.parameter_values.get(scope, {}).copy()

if __name__ == '__main__':
    # Example Usage (for demonstration and basic manual testing)
    print("--- Parameter Manager Demonstration ---")
    manager = ParameterManager()

    # 1. Define parameters for a 'rendering' scope
    print("\n1. Defining parameters for scope 'rendering'...")
    render_params = [
        IntParameter("width", "Render width in pixels", default_value=1920, min_value=1),
        IntParameter("height", "Render height in pixels", default_value=1080, min_value=1),
        FloatParameter("brightness", "Image brightness", default_value=0.5, min_value=0.0, max_value=1.0),
        StringParameter("engine", "Rendering engine", default_value="cycles", choices=["cycles", "eevee"]),
        BoolParameter("use_denoising", "Enable denoising", default_value=True)
    ]
    try:
        manager.define_parameters("rendering", render_params)
        print("Parameters defined for 'rendering'.")
    except Exception as e:
        print(f"Error defining parameters: {e}")

    # 2. Define parameters for a 'simulation' scope
    print("\n2. Defining parameters for scope 'simulation'...")
    sim_params = [
        IntParameter("time_steps", "Number of simulation steps", default_value=100, min_value=10),
        FloatParameter("gravity", "Gravitational constant", default_value=-9.81),
        StringParameter("solver", "Physics solver type", default_value="rk4")
    ]
    manager.define_parameters("simulation", sim_params)
    print("Parameters defined for 'simulation'.")

    # 3. Get parameter definitions for 'rendering' scope
    print("\n3. Getting parameter definitions for 'rendering':")
    render_defs = manager.get_parameter_definitions("rendering")
    for name, details in render_defs.items():
        print(f"  - {name}: {details}")

    # 4. Get all parameter values for 'rendering' scope (should be defaults)
    print("\n4. Getting all parameter values for 'rendering' (defaults):")
    render_values = manager.get_all_parameter_values("rendering")
    for name, value in render_values.items():
        print(f"  - {name}: {value} (type: {type(value).__name__})")

    # 5. Set some parameters in 'rendering' scope
    print("\n5. Setting parameters for 'rendering':")
    try:
        manager.set_parameter("rendering", "width", 1280)
        print("  - Set 'width' to 1280")
        manager.set_parameter("rendering", "brightness", 0.75) # Valid float
        print("  - Set 'brightness' to 0.75")
        manager.set_parameter("rendering", "engine", "eevee")
        print("  - Set 'engine' to 'eevee'")
        # manager.set_parameter("rendering", "brightness", 7) # Valid int for float param
        # print("  - Set 'brightness' to 7 (int)")
    except (ValueError, KeyError) as e:
        print(f"  Error setting parameter: {e}")

    # 6. Get individual parameters from 'rendering'
    print("\n6. Getting individual parameters from 'rendering':")
    print(f"  - Width: {manager.get_parameter('rendering', 'width')}")
    print(f"  - Brightness: {manager.get_parameter('rendering', 'brightness')}")
    print(f"  - Engine: {manager.get_parameter('rendering', 'engine')}")

    # 7. Attempt to set invalid values
    print("\n7. Attempting to set invalid values for 'rendering':")
    try:
        manager.set_parameter("rendering", "width", -100) # Invalid: below min_value
    except ValueError as e:
        print(f"  Error (expected): {e}")
    try:
        manager.set_parameter("rendering", "brightness", 2.0) # Invalid: above max_value
    except ValueError as e:
        print(f"  Error (expected): {e}")
    try:
        manager.set_parameter("rendering", "engine", "unreal") # Invalid: not in choices
    except ValueError as e:
        print(f"  Error (expected): {e}")
    try:
        manager.set_parameter("rendering", "use_denoising", "maybe") # Invalid: not a bool
    except ValueError as e:
        print(f"  Error (expected): {e}")
    try:
        manager.set_parameter("rendering", "non_existent_param", "value") # Invalid: param not defined
    except KeyError as e:
        print(f"  Error (expected): {e}")

    # 8. Get all parameter values for 'simulation' scope
    print("\n8. Getting all parameter values for 'simulation' (defaults):")
    sim_values = manager.get_all_parameter_values("simulation")
    for name, value in sim_values.items():
        print(f"  - {name}: {value}")

    # 9. Get parameter definitions for non-existent scope
    print("\n9. Getting parameter definitions for 'non_existent_scope':")
    non_existent_defs = manager.get_parameter_definitions("non_existent_scope")
    print(f"  Definitions: {non_existent_defs}")

    # 10. Get all parameter values for non-existent scope
    print("\n10. Getting all parameter values for 'non_existent_scope':")
    non_existent_values = manager.get_all_parameter_values("non_existent_scope")
    print(f"  Values: {non_existent_values}")

    # 11. Test FloatParameter with integer input for set_parameter
    print("\n11. Testing FloatParameter with integer input for set_parameter:")
    manager.set_parameter("rendering", "brightness", 0) # int value for float parameter
    print(f"  - Brightness after setting with int 0: {manager.get_parameter('rendering', 'brightness')} (type: {type(manager.get_parameter('rendering', 'brightness')).__name__})")
    self_brightness_details = manager.get_parameter_definitions("rendering")["brightness"]
    print(f"  - Brightness default value type from get_details: {type(self_brightness_details['default_value']).__name__}")


    print("\n--- Demonstration Finished ---")
