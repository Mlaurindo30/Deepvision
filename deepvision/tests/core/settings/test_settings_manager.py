import unittest
import os
import json
from deepvision.deepvision_core.settings.settings_manager import SettingsManager

class TestSettingsManager(unittest.TestCase):
    """
    Unit tests for the SettingsManager class.
    """

    def setUp(self):
        """
        Set up test environment.
        Creates temporary config files for testing.
        """
        self.test_config_file = "test_config.json"
        self.invalid_config_file = "invalid_config.json"
        self.non_existent_config_file = "non_existent_config.json"
        self.new_save_config_file = "new_save_config.json"

        # Clean up any old test files before starting
        self._cleanup_files()

        # Create a valid config file
        with open(self.test_config_file, 'w') as f:
            json.dump({"general_settings": {"theme": "dark_from_file"}, "user_key": "user_value"}, f, indent=4)

        # Create an invalid config file
        with open(self.invalid_config_file, 'w') as f:
            f.write('{"invalid_json": "value",') # Trailing comma makes it invalid

    def tearDown(self):
        """
        Clean up test environment.
        Removes temporary config files created during tests.
        """
        self._cleanup_files()

    def _cleanup_files(self):
        """Helper method to remove all known temporary files."""
        files_to_remove = [
            self.test_config_file,
            self.invalid_config_file,
            self.non_existent_config_file,
            self.new_save_config_file,
            "config.json" # From example in SettingsManager
        ]
        for f_path in files_to_remove:
            if os.path.exists(f_path):
                os.remove(f_path)
            # Also remove potential files created by SettingsManager's __main__ example
            if os.path.exists(os.path.join(os.getcwd(), f_path)): # In case paths were relative
                 os.remove(os.path.join(os.getcwd(), f_path))


    def test_load_default_settings_on_missing_file(self):
        """
        Tests that default settings are loaded when the config file is missing.
        """
        if os.path.exists(self.non_existent_config_file):
            os.remove(self.non_existent_config_file) # Ensure it's missing

        manager = SettingsManager(self.non_existent_config_file)
        self.assertEqual(manager.settings, manager.default_settings)
        self.assertEqual(manager.get_setting("general_settings.theme"), manager.default_settings["general_settings"]["theme"])
        self.assertEqual(manager.get_setting("general_settings.language"), manager.default_settings["general_settings"]["language"])
        self.assertTrue(manager.get_setting("notifications.enabled"))

    def test_load_from_existing_file(self):
        """
        Tests loading settings from an existing and valid JSON config file.
        """
        manager = SettingsManager(self.test_config_file)
        self.assertEqual(manager.get_setting("general_settings.theme"), "dark_from_file")
        self.assertEqual(manager.get_setting("user_key"), "user_value")
        # Check that a key only in default_settings is still accessible
        self.assertEqual(manager.get_setting("general_settings.language"), manager.default_settings["general_settings"]["language"])
        self.assertTrue(manager.get_setting("notifications.enabled")) # From default

    def test_get_setting(self):
        """
        Tests the get_setting method for various scenarios.
        """
        manager = SettingsManager(self.test_config_file) # Loads test_config.json

        # Key from file
        self.assertEqual(manager.get_setting("user_key"), "user_value")
        # Nested key from file
        self.assertEqual(manager.get_setting("general_settings.theme"), "dark_from_file")
        # Key from default settings (not in test_config.json)
        self.assertEqual(manager.get_setting("general_settings.language"), manager.default_settings["general_settings"]["language"])
        # Non-existent key with fallback
        self.assertEqual(manager.get_setting("non.existent.key", "fallback_value"), "fallback_value")
        # Non-existent key without fallback
        self.assertIsNone(manager.get_setting("another.non.existent.key"))
        # Non-existent nested key
        self.assertIsNone(manager.get_setting("general_settings.non_existent_sub_key"))
        self.assertEqual(manager.get_setting("general_settings.non_existent_sub_key", "sub_fallback"), "sub_fallback")

    def test_set_setting(self):
        """
        Tests the set_setting method for new and existing keys, including nested ones.
        """
        manager = SettingsManager(self.non_existent_config_file) # Start with defaults

        # Set new first-level key
        manager.set_setting("my_key", "my_value")
        self.assertEqual(manager.get_setting("my_key"), "my_value")

        # Set new nested key
        manager.set_setting("level1.level2.deep_key", "deep_value")
        self.assertEqual(manager.get_setting("level1.level2.deep_key"), "deep_value")
        self.assertIsInstance(manager.settings.get("level1"), dict)
        self.assertIsInstance(manager.settings.get("level1", {}).get("level2"), dict)

        # Modify existing key (from defaults)
        manager.set_setting("general_settings.theme", "ocean_blue")
        self.assertEqual(manager.get_setting("general_settings.theme"), "ocean_blue")

        # Modify a key that was just set
        manager.set_setting("my_key", "updated_value")
        self.assertEqual(manager.get_setting("my_key"), "updated_value")

    def test_save_and_reload_settings(self):
        """
        Tests saving settings to a file and then reloading them in a new instance.
        """
        manager1 = SettingsManager(self.new_save_config_file)
        if os.path.exists(self.new_save_config_file): # ensure clean start
            os.remove(self.new_save_config_file)

        manager1.set_setting("user.preference.option", True)
        manager1.set_setting("user.name", "Test User")
        manager1.set_setting("general_settings.theme", "test_theme_save")
        manager1.save_settings()

        self.assertTrue(os.path.exists(self.new_save_config_file))

        manager2 = SettingsManager(self.new_save_config_file)
        self.assertEqual(manager2.get_setting("user.preference.option"), True)
        self.assertEqual(manager2.get_setting("user.name"), "Test User")
        self.assertEqual(manager2.get_setting("general_settings.theme"), "test_theme_save")
        # Ensure default settings that were not overwritten are still present
        self.assertEqual(manager2.get_setting("general_settings.language"), manager1.default_settings["general_settings"]["language"])

    def test_load_invalid_json_file(self):
        """
        Tests that loading an invalid JSON file results in default settings being used.
        """
        manager = SettingsManager(self.invalid_config_file)
        # Settings should revert to defaults due to JSONDecodeError during load
        self.assertEqual(manager.settings, manager.default_settings)
        self.assertEqual(manager.get_setting("general_settings.theme"), manager.default_settings["general_settings"]["theme"])
        # Ensure no exception was raised during __init__ (it should be handled)

    def test_save_creates_new_file(self):
        """
        Tests that save_settings creates a new file if it doesn't exist.
        """
        save_path = "completely_new_config.json" # Use a unique name for this test
        self.addCleanup(lambda: os.remove(save_path) if os.path.exists(save_path) else None) # Ensure cleanup

        if os.path.exists(save_path):
            os.remove(save_path) # Ensure it's missing before test

        manager = SettingsManager(save_path)
        manager.set_setting("feature.enabled", True)
        manager.set_setting("feature.level", 5)
        manager.save_settings()

        self.assertTrue(os.path.exists(save_path))

        # Optional: Load and verify content
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        expected_data = manager.default_settings.copy() # Start with defaults
        # Manually apply the changes made in the test to the expected_data
        expected_data["feature"] = {"enabled": True, "level": 5}
        
        # Need to reconstruct the expected settings based on how SettingsManager merges
        # For this specific test, we started with defaults, then added 'feature'
        # So, the saved data should be defaults + the new 'feature' settings.
        # The manager.settings would reflect this exact state.
        self.assertEqual(saved_data, manager.settings)
        self.assertEqual(saved_data.get("feature", {}).get("enabled"), True)
        self.assertEqual(saved_data.get("feature", {}).get("level"), 5)
        self.assertEqual(saved_data.get("general_settings", {}).get("theme"), manager.default_settings["general_settings"]["theme"])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
