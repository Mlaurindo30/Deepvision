import unittest
from deepvision.deepvision_core.parameters.parameter_manager import (
    BaseParameter, IntParameter, FloatParameter, StringParameter, BoolParameter, ParameterManager
)

class TestParameterTypes(unittest.TestCase):
    """
    Tests for individual parameter type classes (IntParameter, FloatParameter, etc.).
    """

    def test_int_parameter_valid_values(self):
        """Test IntParameter with valid values and constraints."""
        param = IntParameter("test_int", "Test Int", 0, min_value=-10, max_value=10)
        self.assertTrue(param.validate(5))
        self.assertTrue(param.validate(-10))
        self.assertTrue(param.validate(10))
        self.assertEqual(param.default_value, 0)

    def test_int_parameter_invalid_values(self):
        """Test IntParameter with invalid values, checking for ValueError."""
        param = IntParameter("test_int", default_value=0, min_value=0, max_value=100)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            param.validate("not_an_int")
        with self.assertRaisesRegex(ValueError, "less than minimum 0"):
            param.validate(-1)
        with self.assertRaisesRegex(ValueError, "greater than maximum 100"):
            param.validate(101)

    def test_int_parameter_get_details(self):
        """Test IntParameter get_details method."""
        param = IntParameter("num_items", "Number of items", 10, min_value=1, max_value=50)
        details = param.get_details()
        expected_details = {
            "name": "num_items", "description": "Number of items", "type": "int",
            "default_value": 10, "min_value": 1, "max_value": 50
        }
        self.assertEqual(details, expected_details)
        
        param_no_min_max = IntParameter("simple_int", default_value=5)
        details_no_min_max = param_no_min_max.get_details()
        expected_no_min_max = {
            "name": "simple_int", "description": "", "type": "int", "default_value": 5
        }
        self.assertEqual(details_no_min_max, expected_no_min_max)


    def test_float_parameter_valid_values(self):
        """Test FloatParameter with valid values and constraints."""
        param = FloatParameter("test_float", "Test Float", 0.0, min_value=-1.0, max_value=1.0)
        self.assertTrue(param.validate(0.5))
        self.assertTrue(param.validate(-1.0))
        self.assertTrue(param.validate(1)) # Accepts int
        self.assertEqual(param.default_value, 0.0)

    def test_float_parameter_invalid_values(self):
        """Test FloatParameter with invalid values, checking for ValueError."""
        param = FloatParameter("test_float", default_value=0.0, min_value=0.0, max_value=10.0)
        with self.assertRaisesRegex(ValueError, "must be a float or an integer"):
            param.validate("not_a_float")
        with self.assertRaisesRegex(ValueError, "less than minimum 0.0"):
            param.validate(-0.1)
        with self.assertRaisesRegex(ValueError, "greater than maximum 10.0"):
            param.validate(10.001)

    def test_float_parameter_accepts_int(self):
        """Test FloatParameter specifically for accepting integer values."""
        param = FloatParameter("float_accepts_int", default_value=1.0, min_value=0, max_value=10)
        self.assertTrue(param.validate(5)) # int value
        self.assertTrue(param.validate(5.0)) # float value
        self.assertEqual(param.default_value, 1.0)
        
        # Test default value conversion
        param_int_default = FloatParameter("float_int_default", default_value=5)
        self.assertIsInstance(param_int_default.default_value, float)
        self.assertEqual(param_int_default.default_value, 5.0)


    def test_float_parameter_get_details(self):
        """Test FloatParameter get_details method."""
        param = FloatParameter("rate", "Learning rate", 0.001, min_value=0.0, max_value=1.0)
        details = param.get_details()
        expected_details = {
            "name": "rate", "description": "Learning rate", "type": "float",
            "default_value": 0.001, "min_value": 0.0, "max_value": 1.0
        }
        self.assertEqual(details, expected_details)

    def test_string_parameter_valid_values(self):
        """Test StringParameter with valid values and constraints."""
        param = StringParameter("test_str", "Test String", "default", min_length=2, max_length=10, choices=["default", "option1", "option2"])
        self.assertTrue(param.validate("option1"))
        self.assertTrue(param.validate("default"))
        self.assertEqual(param.default_value, "default")

    def test_string_parameter_invalid_values(self):
        """Test StringParameter with invalid values, checking for ValueError."""
        param = StringParameter("test_str", default_value="abc", min_length=3, max_length=5, choices=["abc", "defg"])
        with self.assertRaisesRegex(ValueError, "must be a string"):
            param.validate(123)
        with self.assertRaisesRegex(ValueError, "less than minimum 3"):
            param.validate("ab")
        with self.assertRaisesRegex(ValueError, "greater than maximum 5"):
            param.validate("abcdef")
        with self.assertRaisesRegex(ValueError, "not in allowed choices"):
            param.validate("xyz")

    def test_string_parameter_get_details(self):
        """Test StringParameter get_details method."""
        param = StringParameter("name", "User name", "guest", min_length=3, max_length=20, choices=["guest", "admin", "user"])
        details = param.get_details()
        expected_details = {
            "name": "name", "description": "User name", "type": "string",
            "default_value": "guest", "min_length": 3, "max_length": 20, "choices": ["guest", "admin", "user"]
        }
        self.assertEqual(details, expected_details)
        
        param_simple = StringParameter("simple_str", default_value="hello")
        details_simple = param_simple.get_details()
        expected_simple = {
            "name": "simple_str", "description": "", "type": "string", "default_value": "hello"
        }
        self.assertEqual(details_simple, expected_simple)

    def test_bool_parameter_valid_values(self):
        """Test BoolParameter with valid values."""
        param = BoolParameter("test_bool", "Test Bool", False)
        self.assertTrue(param.validate(True))
        self.assertTrue(param.validate(False))
        self.assertEqual(param.default_value, False)

    def test_bool_parameter_invalid_values(self):
        """Test BoolParameter with invalid values, checking for ValueError."""
        param = BoolParameter("test_bool", default_value=True)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            param.validate("not_a_bool")
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            param.validate(0)

    def test_bool_parameter_get_details(self):
        """Test BoolParameter get_details method."""
        param = BoolParameter("is_enabled", "Is it enabled?", True)
        details = param.get_details()
        expected_details = {
            "name": "is_enabled", "description": "Is it enabled?", "type": "bool", "default_value": True
        }
        self.assertEqual(details, expected_details)

    def test_base_parameter_default_behavior(self):
        """Test BaseParameter default behavior."""
        param = BaseParameter("base_param", "Base description", "base_default")
        self.assertTrue(param.validate("any_value")) # Base validate always true
        self.assertEqual(param.default_value, "base_default")
        details = param.get_details()
        expected_details = {
            "name": "base_param", "description": "Base description", "type": "base", "default_value": "base_default"
        }
        self.assertEqual(details, expected_details)


class TestParameterManager(unittest.TestCase):
    """
    Tests for the ParameterManager class.
    """

    def setUp(self):
        """Instantiate ParameterManager and define some example parameters."""
        self.pm = ParameterManager()
        self.scope_render = "rendering"
        self.scope_sim = "simulation"

        self.render_params = [
            IntParameter("width", "Render width", 1920, min_value=100),
            FloatParameter("brightness", "Image brightness", 0.5, min_value=0.0, max_value=1.0),
            StringParameter("engine", "Render engine", "cycles", choices=["cycles", "eevee"]),
            BoolParameter("denoise", "Enable denoising", False)
        ]
        self.sim_params = [
            IntParameter("steps", "Simulation steps", 100, min_value=1),
            FloatParameter("gravity", "Gravity force", -9.81),
        ]

    def test_define_parameters_and_initial_values(self):
        """Test defining parameters and checking initial (default) values."""
        self.pm.define_parameters(self.scope_render, self.render_params)

        self.assertIn(self.scope_render, self.pm.parameter_definitions)
        self.assertIn(self.scope_render, self.pm.parameter_values)

        for param_def in self.render_params:
            self.assertIn(param_def.name, self.pm.parameter_definitions[self.scope_render])
            self.assertIsInstance(self.pm.parameter_definitions[self.scope_render][param_def.name], BaseParameter)
            self.assertEqual(self.pm.parameter_values[self.scope_render][param_def.name], param_def.default_value)
            # Check if FloatParameter stores int default as float
            if isinstance(param_def, FloatParameter) and isinstance(param_def.default_value, int):
                 self.assertIsInstance(self.pm.parameter_values[self.scope_render][param_def.name], float)

    def test_define_parameters_invalid_definition_type(self):
        """Test defining parameters with an invalid definition type."""
        with self.assertRaisesRegex(TypeError, "Invalid parameter definition type"):
            self.pm.define_parameters("invalid_scope", [IntParameter("good"), "not_a_param_def"])


    def test_get_parameter_definitions(self):
        """Test retrieving parameter definitions for a scope."""
        self.pm.define_parameters(self.scope_render, self.render_params)
        definitions = self.pm.get_parameter_definitions(self.scope_render)

        self.assertEqual(len(definitions), len(self.render_params))
        for param_def in self.render_params:
            self.assertIn(param_def.name, definitions)
            self.assertEqual(definitions[param_def.name]["name"], param_def.name)
            self.assertEqual(definitions[param_def.name]["type"], param_def.param_type)
            self.assertEqual(definitions[param_def.name]["default_value"], param_def.default_value)

        # Test with a non-existent scope
        self.assertEqual(self.pm.get_parameter_definitions("non_existent_scope"), {})

    def test_set_and_get_parameter_valid(self):
        """Test setting and getting parameters with valid values."""
        self.pm.define_parameters(self.scope_render, self.render_params)

        self.pm.set_parameter(self.scope_render, "width", 1280)
        self.assertEqual(self.pm.get_parameter(self.scope_render, "width"), 1280)

        self.pm.set_parameter(self.scope_render, "brightness", 0.75)
        self.assertEqual(self.pm.get_parameter(self.scope_render, "brightness"), 0.75)
        
        self.pm.set_parameter(self.scope_render, "brightness", 0) # int for float
        self.assertEqual(self.pm.get_parameter(self.scope_render, "brightness"), 0.0)
        self.assertIsInstance(self.pm.get_parameter(self.scope_render, "brightness"), float)


        self.pm.set_parameter(self.scope_render, "engine", "eevee")
        self.assertEqual(self.pm.get_parameter(self.scope_render, "engine"), "eevee")

        self.pm.set_parameter(self.scope_render, "denoise", True)
        self.assertEqual(self.pm.get_parameter(self.scope_render, "denoise"), True)

    def test_set_parameter_invalid_value_raises_valueerror(self):
        """Test that setting an invalid parameter value raises ValueError and doesn't change the value."""
        self.pm.define_parameters(self.scope_render, self.render_params)

        # Test IntParameter invalid value
        original_width = self.pm.get_parameter(self.scope_render, "width")
        with self.assertRaisesRegex(ValueError, "less than minimum 100"):
            self.pm.set_parameter(self.scope_render, "width", 50)
        self.assertEqual(self.pm.get_parameter(self.scope_render, "width"), original_width) # Check original value

        # Test FloatParameter invalid value
        original_brightness = self.pm.get_parameter(self.scope_render, "brightness")
        with self.assertRaisesRegex(ValueError, "greater than maximum 1.0"):
            self.pm.set_parameter(self.scope_render, "brightness", 1.5)
        self.assertEqual(self.pm.get_parameter(self.scope_render, "brightness"), original_brightness)

        # Test StringParameter invalid choice
        original_engine = self.pm.get_parameter(self.scope_render, "engine")
        with self.assertRaisesRegex(ValueError, "not in allowed choices"):
            self.pm.set_parameter(self.scope_render, "engine", "vray")
        self.assertEqual(self.pm.get_parameter(self.scope_render, "engine"), original_engine)

        # Test BoolParameter invalid type
        original_denoise = self.pm.get_parameter(self.scope_render, "denoise")
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.pm.set_parameter(self.scope_render, "denoise", "True_str")
        self.assertEqual(self.pm.get_parameter(self.scope_render, "denoise"), original_denoise)

    def test_get_parameter_unknown_scope_or_name_raises_keyerror(self):
        """Test that getting a parameter from an unknown scope or with an unknown name raises KeyError."""
        self.pm.define_parameters(self.scope_render, self.render_params)

        with self.assertRaisesRegex(KeyError, "Scope 'unknown_scope' not found"):
            self.pm.get_parameter("unknown_scope", "width")

        with self.assertRaisesRegex(KeyError, "Parameter 'unknown_param' not found in scope 'rendering'"):
            self.pm.get_parameter(self.scope_render, "unknown_param")

    def test_set_parameter_unknown_scope_or_name_raises_keyerror(self):
        """Test that setting a parameter for an unknown scope or with an unknown name raises KeyError."""
        self.pm.define_parameters(self.scope_render, self.render_params)

        with self.assertRaisesRegex(KeyError, "Scope 'unknown_scope' not found. Define parameters first."):
            self.pm.set_parameter("unknown_scope", "width", 1000)

        with self.assertRaisesRegex(KeyError, "Parameter 'unknown_param' not defined in scope 'rendering'. Define parameters first."):
            self.pm.set_parameter(self.scope_render, "unknown_param", 1000)

    def test_get_all_parameter_values(self):
        """Test retrieving all parameter values for a scope."""
        self.pm.define_parameters(self.scope_render, self.render_params)
        self.pm.set_parameter(self.scope_render, "width", 1280)
        self.pm.set_parameter(self.scope_render, "denoise", True)

        values = self.pm.get_all_parameter_values(self.scope_render)
        self.assertEqual(len(values), len(self.render_params))
        self.assertEqual(values["width"], 1280)
        self.assertEqual(values["brightness"], 0.5) # Default
        self.assertEqual(values["engine"], "cycles") # Default
        self.assertEqual(values["denoise"], True)

        # Test with a non-existent scope
        self.assertEqual(self.pm.get_all_parameter_values("non_existent_scope"), {})

    def test_parameter_scoping_no_interference(self):
        """Test that parameters with the same name in different scopes do not interfere."""
        threshold_param_render = IntParameter("threshold", "Render threshold", 10, min_value=0)
        threshold_param_sim = IntParameter("threshold", "Simulation threshold", 5, min_value=0)

        self.pm.define_parameters(self.scope_render, [threshold_param_render])
        self.pm.define_parameters(self.scope_sim, [threshold_param_sim])

        # Check initial default values
        self.assertEqual(self.pm.get_parameter(self.scope_render, "threshold"), 10)
        self.assertEqual(self.pm.get_parameter(self.scope_sim, "threshold"), 5)

        # Set values
        self.pm.set_parameter(self.scope_render, "threshold", 20)
        self.pm.set_parameter(self.scope_sim, "threshold", 3)

        # Verify they are independent
        self.assertEqual(self.pm.get_parameter(self.scope_render, "threshold"), 20)
        self.assertEqual(self.pm.get_parameter(self.scope_sim, "threshold"), 3)
        
        # Verify definitions are also distinct
        render_defs = self.pm.get_parameter_definitions(self.scope_render)
        sim_defs = self.pm.get_parameter_definitions(self.scope_sim)
        self.assertEqual(render_defs["threshold"]["default_value"], 10)
        self.assertEqual(sim_defs["threshold"]["default_value"], 5)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
