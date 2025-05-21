import json
from typing import Dict, Any, Callable, List, Tuple, Optional

# Simulated layout data for testing if actual files are not available
SIMULATED_SWAPPER_LAYOUT_DATA = {
    "swapper_options": { # Category
        "face_margin_Slider": { # Parameter name
            "widget_type": "Slider",
            "label": "Face Margin",
            "tooltip": "Adjust the margin around the detected face for swapping.",
            "min": 0,
            "max": 100,
            "step": 1,
            "default": 20, # Default value
            "category": "swapper_options"
        },
        "face_blend_DecimalSlider": {
            "widget_type": "DecimalSlider",
            "label": "Face Blend",
            "tooltip": "Blending amount for the swapped face.",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "default": 0.75,
            "category": "swapper_options"
        },
        "enable_enhancements_Toggle": {
            "widget_type": "Toggle",
            "label": "Enable Enhancements",
            "tooltip": "Enable post-processing enhancements.",
            "default": True,
            "category": "swapper_options"
        },
        "upscaler_model_Selection": {
            "widget_type": "Selection",
            "label": "Upscaler Model",
            "tooltip": "Choose the upscaler model to use.",
            "options": ["None", "RealESRGAN", "GFPGAN"],
            "default": "RealESRGAN",
            "category": "swapper_options"
        },
        "another_int_param_Slider": { # Ensure unique names even if label is same
            "widget_type": "Slider",
            "label": "Some Integer",
            "tooltip": "Another integer parameter.",
            "min": -50,
            "max": 50,
            "step": 5,
            "default": 0,
            "category": "swapper_options"
        }
    },
    "enhancement_settings": { # Another Category
        "color_correction_Toggle": {
            "widget_type": "Toggle",
            "label": "Color Correction",
            "tooltip": "Enable color correction for the swapped face.",
            "default": True,
            "category": "enhancement_settings"
        },
        "sharpening_amount_DecimalSlider": {
            "widget_type": "DecimalSlider",
            "label": "Sharpening Amount",
            "tooltip": "Amount of sharpening to apply.",
            "min": 0.0,
            "max": 5.0,
            "step": 0.1,
            "default": 0.5,
            "category": "enhancement_settings"
        }
    }
}


class ParameterManager:
    def __init__(self):
        self._parameter_definitions: Dict[str, Dict[str, Any]] = {}
        self._parameter_values: Dict[str, Any] = {}
        self._observers: List[Callable[[str, Any], None]] = []

    def _infer_type_and_constraints(self, name: str, widget_attrs: Dict[str, Any]) -> Tuple[type, Dict[str, Any]]:
        """Infers parameter type and constraints from widget attributes."""
        widget_type = widget_attrs.get("widget_type")
        constraints = {}
        param_type: type = str # Default type

        if widget_type == "Toggle" or name.endswith("Toggle"):
            param_type = bool
        elif widget_type == "Selection" or name.endswith("Selection"):
            param_type = str
            constraints["options"] = widget_attrs.get("options", [])
        elif widget_type == "DecimalSlider" or name.endswith("DecimalSlider"):
            param_type = float
            constraints["min"] = widget_attrs.get("min")
            constraints["max"] = widget_attrs.get("max")
            constraints["step"] = widget_attrs.get("step")
        elif widget_type == "Slider" or name.endswith("Slider"):
            param_type = int
            constraints["min"] = widget_attrs.get("min")
            constraints["max"] = widget_attrs.get("max")
            constraints["step"] = widget_attrs.get("step")
        elif widget_type == "TextInput" or name.endswith("TextInput"): # Assuming TextInput exists
             param_type = str
        # Add more inferences as needed

        # Ensure min/max/step are not None if they are relevant
        if param_type in [int, float]:
            for key in ["min", "max", "step"]:
                if constraints.get(key) is None and widget_attrs.get(key) is not None:
                     constraints[key] = widget_attrs.get(key)
        return param_type, constraints

    def register_parameters_from_layout_data(self, layout_data: Dict):
        """Populates parameter definitions and values from layout data."""
        for category_name, category_widgets in layout_data.items():
            for param_name, widget_attrs in category_widgets.items():
                param_type, constraints = self._infer_type_and_constraints(param_name, widget_attrs)
                
                default_value = widget_attrs.get("default")
                # Validate default_value type if possible (basic check)
                if default_value is not None and not isinstance(default_value, param_type):
                    try:
                        if param_type is bool and isinstance(default_value, str): # "true"/"false"
                             default_value = default_value.lower() == "true"
                        else:
                             default_value = param_type(default_value)
                    except ValueError:
                        print(f"Warning: Default value {default_value} for {param_name} does not match inferred type {param_type}. Using type's default.")
                        default_value = param_type() # bool() is False, int() is 0, etc.
                
                self._parameter_definitions[param_name] = {
                    "name": param_name,
                    "default": default_value,
                    "type": param_type,
                    "constraints": constraints,
                    "category": widget_attrs.get("category", category_name), # Use widget's category if specified, else use the group key
                    "label": widget_attrs.get("label", param_name),
                    "tooltip": widget_attrs.get("tooltip", "")
                }
                self._parameter_values[param_name] = default_value
        self._notify_all() # Notify observers about the new parameters

    def get_parameter_value(self, name: str) -> Any:
        """Gets the current value of a parameter."""
        if name not in self._parameter_values:
            # print(f"Warning: Parameter {name} not found. Returning None.")
            return None
        return self._parameter_values.get(name)

    def set_parameter_value(self, name: str, value: Any) -> bool:
        """Sets the value of a parameter with validation and clamping."""
        if name not in self._parameter_definitions:
            # print(f"Error: Parameter {name} not defined.")
            return False

        definition = self._parameter_definitions[name]
        param_type = definition["type"]
        constraints = definition["constraints"]

        # Type validation
        if not isinstance(value, param_type):
            try:
                # Attempt type coercion for common cases (e.g. string "123" to int 123)
                if param_type is bool and isinstance(value, str): # "true"/"false"
                    value = value.lower() == "true"
                else:
                    value = param_type(value)
                if not isinstance(value, param_type): # Check again after coercion
                    raise TypeError()
            except (TypeError, ValueError):
                # print(f"Error: Value for {name} has incorrect type. Expected {param_type}, got {type(value)}.")
                return False
        
        # Constraint validation
        if param_type in [int, float]:
            min_val = constraints.get("min")
            max_val = constraints.get("max")
            # Clamping for numeric types
            if min_val is not None and value < min_val:
                value = min_val
            if max_val is not None and value > max_val:
                value = max_val
        elif param_type == str and "options" in constraints:
            if value not in constraints["options"]:
                # print(f"Error: Value '{value}' for {name} is not in allowed options {constraints['options']}.")
                return False
        
        self._parameter_values[name] = value
        self._notify(name, value)
        return True

    def get_parameter_definition(self, name: str) -> Optional[Dict[str, Any]]:
        """Gets the definition of a parameter."""
        return self._parameter_definitions.get(name)

    def get_all_current_values(self) -> Dict[str, Any]:
        """Gets a dictionary of all current parameter values."""
        return self._parameter_values.copy()

    def get_parameters_for_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Gets definitions for all parameters in a specific category."""
        return {
            name: definition
            for name, definition in self._parameter_definitions.items()
            if definition["category"] == category
        }

    def load_parameters_profile(self, filepath: str) -> bool:
        """Loads parameter values from a JSON file."""
        try:
            if not os.path.exists(filepath):
                # print(f"Error: Profile file {filepath} not found.")
                return False
            with open(filepath, 'r') as f:
                loaded_values = json.load(f)
            
            success = True
            for name, value in loaded_values.items():
                if name in self._parameter_definitions: # Only load values for known parameters
                    if not self.set_parameter_value(name, value):
                        # print(f"Warning: Could not set value for {name} from profile {filepath}. Using current or default.")
                        success = False # Partial success
                # else:
                    # print(f"Warning: Parameter {name} from profile {filepath} not recognized. Skipping.")
            # self._notify_all() # set_parameter_value already notifies individually
            return success
        except json.JSONDecodeError:
            # print(f"Error: Could not decode JSON from profile file {filepath}.")
            return False
        except Exception as e:
            # print(f"An error occurred while loading profile {filepath}: {e}")
            return False

    def save_parameters_profile(self, filepath: str) -> bool:
        """Saves current parameter values to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self._parameter_values, f, indent=4)
            return True
        except Exception as e:
            # print(f"An error occurred while saving profile {filepath}: {e}")
            return False

    def reset_to_defaults(self):
        """Resets all parameters to their default values."""
        for name, definition in self._parameter_definitions.items():
            self._parameter_values[name] = definition["default"]
        self._notify_all()

    def reset_category_to_defaults(self, category: str):
        """Resets all parameters in a specific category to their default values."""
        for name, definition in self._parameter_definitions.items():
            if definition["category"] == category:
                self._parameter_values[name] = definition["default"]
                self._notify(name, definition["default"]) # Notify for each changed param in category

    def subscribe(self, observer: Callable[[str, Any], None]):
        """Subscribes an observer function to parameter changes."""
        if observer not in self._observers:
            self._observers.append(observer)

    def _notify(self, param_name: str, value: Any):
        """Notifies all observers about a change to a specific parameter."""
        for observer in self._observers:
            try:
                observer(param_name, value)
            except Exception as e:
                print(f"Error notifying observer {observer}: {e}")
    
    def _notify_all(self):
        """Notifies observers about changes to all parameters (e.g., after reset or load)."""
        for name, value in self._parameter_values.items():
             self._notify(name, value)

if __name__ == '__main__':
    import os # For file operations in example

    # --- Example Usage ---
    param_manager = ParameterManager()

    # Observer example
    def simple_observer(param_name: str, value: Any):
        print(f"OBSERVER: Parameter '{param_name}' changed to '{value}'")

    param_manager.subscribe(simple_observer)

    print("Registering parameters from SIMULATED_SWAPPER_LAYOUT_DATA...")
    param_manager.register_parameters_from_layout_data(SIMULATED_SWAPPER_LAYOUT_DATA)
    
    print("\n--- Initial Parameter Values ---")
    print(json.dumps(param_manager.get_all_current_values(), indent=2))

    print("\n--- Getting a specific parameter definition (face_blend_DecimalSlider) ---")
    print(json.dumps(param_manager.get_parameter_definition("face_blend_DecimalSlider"), indent=2))

    print("\n--- Setting Parameter Values ---")
    print("Setting 'face_margin_Slider' to 50:", param_manager.set_parameter_value("face_margin_Slider", 50))
    print("Setting 'face_blend_DecimalSlider' to 0.9 (valid):", param_manager.set_parameter_value("face_blend_DecimalSlider", 0.9))
    print("Setting 'face_blend_DecimalSlider' to 1.5 (invalid, should clamp to 1.0):", param_manager.set_parameter_value("face_blend_DecimalSlider", 1.5))
    print("Value after clamping:", param_manager.get_parameter_value("face_blend_DecimalSlider"))
    print("Setting 'enable_enhancements_Toggle' to False:", param_manager.set_parameter_value("enable_enhancements_Toggle", False))
    print("Setting 'upscaler_model_Selection' to 'GFPGAN' (valid):", param_manager.set_parameter_value("upscaler_model_Selection", "GFPGAN"))
    print("Setting 'upscaler_model_Selection' to 'InvalidModel' (invalid):", param_manager.set_parameter_value("upscaler_model_Selection", "InvalidModel"))
    print("Setting 'non_existent_param' to 100 (invalid):", param_manager.set_parameter_value("non_existent_param", 100))
    
    # Test type coercion
    print("Setting 'face_margin_Slider' to '75' (string, should coerce to int):", param_manager.set_parameter_value("face_margin_Slider", "75"))
    print("Value after coercion:", param_manager.get_parameter_value("face_margin_Slider"))
    print("Setting 'enable_enhancements_Toggle' to 'true' (string, should coerce to bool):", param_manager.set_parameter_value("enable_enhancements_Toggle", "true"))
    print("Value after coercion:", param_manager.get_parameter_value("enable_enhancements_Toggle"))


    print("\n--- Current Values After Some Changes ---")
    print(json.dumps(param_manager.get_all_current_values(), indent=2))

    print("\n--- Parameters for category 'enhancement_settings' ---")
    print(json.dumps(param_manager.get_parameters_for_category("enhancement_settings"), indent=2))

    profile_path = "test_profile.json"
    print(f"\n--- Saving parameters to profile '{profile_path}' ---")
    param_manager.save_parameters_profile(profile_path)

    print("\n--- Resetting 'swapper_options' category to defaults ---")
    param_manager.reset_category_to_defaults("swapper_options")
    print(json.dumps(param_manager.get_all_current_values(), indent=2))
    
    print(f"\n--- Loading parameters from profile '{profile_path}' ---")
    # Modify a value before loading to see the change
    param_manager.set_parameter_value("sharpening_amount_DecimalSlider", 2.0) 
    print("Value of 'sharpening_amount_DecimalSlider' before load:", param_manager.get_parameter_value("sharpening_amount_DecimalSlider"))
    param_manager.load_parameters_profile(profile_path)
    print("Value of 'sharpening_amount_DecimalSlider' after load:", param_manager.get_parameter_value("sharpening_amount_DecimalSlider"))
    print("All values after load:")
    print(json.dumps(param_manager.get_all_current_values(), indent=2))

    print("\n--- Resetting all parameters to defaults ---")
    param_manager.reset_to_defaults()
    print(json.dumps(param_manager.get_all_current_values(), indent=2))

    # Test loading a profile with a non-existent parameter or bad value
    malformed_profile_path = "malformed_profile.json"
    with open(malformed_profile_path, 'w') as f:
        json.dump({"face_margin_Slider": 999, "new_unknown_param": "test", "face_blend_DecimalSlider": "not_a_float"}, f)
    
    print(f"\n--- Loading parameters from malformed profile '{malformed_profile_path}' ---")
    param_manager.load_parameters_profile(malformed_profile_path) # Should clamp face_margin, skip unknown, fail on bad type
    print("Values after loading malformed profile:")
    print(json.dumps(param_manager.get_all_current_values(), indent=2)) # Observe clamping and skips


    # Clean up test files
    if os.path.exists(profile_path):
        os.remove(profile_path)
        print(f"\nCleaned up {profile_path}")
    if os.path.exists(malformed_profile_path):
        os.remove(malformed_profile_path)
        print(f"Cleaned up {malformed_profile_path}")

    print("\n--- Testing get_parameter_value for non-existent parameter ---")
    print("Value for 'does_not_exist':", param_manager.get_parameter_value("does_not_exist"))

    print("\n--- Example of trying to set a value for a parameter not registered ---")
    if not param_manager.set_parameter_value("unregistered_param", 10):
        print("Correctly failed to set 'unregistered_param'")

    # Test default value type mismatch handling during registration
    faulty_layout_data = {
        "test_category": {
            "bad_default_int_Slider": {
                "widget_type": "Slider", "label": "Bad Default Int", "min": 0, "max": 10, "step": 1, "default": "not-an-int", "category": "test_category"
            },
             "bad_default_bool_Toggle": {
                "widget_type": "Toggle", "label": "Bad Default Bool", "default": "maybe", "category": "test_category"
            }
        }
    }
    print("\n--- Registering parameters with faulty default values ---")
    param_manager.register_parameters_from_layout_data(faulty_layout_data)
    print("Value for 'bad_default_int_Slider' (should be 0):", param_manager.get_parameter_value("bad_default_int_Slider"))
    print("Value for 'bad_default_bool_Toggle' (should be False):", param_manager.get_parameter_value("bad_default_bool_Toggle"))
    
    print("\n--- End of Example Usage ---")
