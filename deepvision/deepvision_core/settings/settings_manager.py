import json
import os

class SettingsManager:
    """
    Manages application settings, loading from and saving to a JSON file.
    """
    def __init__(self, config_file_path: str):
        """
        Initializes the SettingsManager.

        Args:
            config_file_path (str): The path to the JSON configuration file.
        """
        self.config_file_path = config_file_path
        self.settings = {}
        self.default_settings = {
            "general_settings": {
                "theme": "light",
                "language": "pt_BR"
            },
            "notifications": {
                "enabled": True
            }
        }
        self._load_settings()

    def _load_settings(self):
        """
        Loads settings from the configuration file.

        If the file doesn't exist or is invalid, default settings are used.
        """
        self.settings = self.default_settings.copy()  # Start with defaults

        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, 'r') as f:
                    file_settings = json.load(f)
                # Deep update self.settings with file_settings
                # This allows overriding specific nested keys without replacing entire parent keys
                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict):
                            d[k] = update_dict(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d
                self.settings = update_dict(self.settings, file_settings)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {self.config_file_path}. Using default settings.")
            except Exception as e:
                print(f"Warning: Error loading {self.config_file_path}: {e}. Using default settings.")
        else:
            print(f"Info: Configuration file {self.config_file_path} not found. Using default settings.")

    def save_settings(self):
        """
        Saves the current settings to the configuration file.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
            with open(self.config_file_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error: Could not save settings to {self.config_file_path}: {e}")

    def get_setting(self, key: str, default_value=None):
        """
        Retrieves a setting value.

        Args:
            key (str): The key of the setting to retrieve (e.g., "general_settings.theme").
            default_value: The value to return if the key is not found.

        Returns:
            The value of the setting, or default_value if not found.
        """
        keys = key.split('.')
        current_level = self.settings
        for k in keys:
            if isinstance(current_level, dict) and k in current_level:
                current_level = current_level[k]
            else:
                return default_value
        return current_level

    def set_setting(self, key: str, value):
        """
        Sets a setting value.

        Args:
            key (str): The key of the setting to set (e.g., "general_settings.theme").
                         Intermediate dictionaries will be created if they don't exist.
            value: The value to set.
        """
        keys = key.split('.')
        current_level = self.settings
        for i, k in enumerate(keys):
            if i == len(keys) - 1:  # Last key
                current_level[k] = value
            else:
                if k not in current_level or not isinstance(current_level[k], dict):
                    current_level[k] = {}
                current_level = current_level[k]

if __name__ == '__main__':
    # Example Usage (for testing purposes, not part of the class itself)
    # Create a dummy config.json for testing if it doesn't exist
    if not os.path.exists('config.json'):
        with open('config.json', 'w') as f:
            json.dump({
                "general_settings": {
                    "theme": "dark_from_file",
                    "language": "en_US_from_file"
                },
                "user_specific": {
                    "user_id": 123
                }
            }, f, indent=4)

    print("--- Initializing SettingsManager with config.json ---")
    manager = SettingsManager('config.json')

    print("\n--- Current Settings (after load) ---")
    print(json.dumps(manager.settings, indent=4))

    print("\n--- Getting Specific Settings ---")
    print(f"Theme: {manager.get_setting('general_settings.theme', 'default_theme')}")
    print(f"Language: {manager.get_setting('general_settings.language')}")
    print(f"Notifications Enabled: {manager.get_setting('notifications.enabled')}")
    print(f"User ID (from file): {manager.get_setting('user_specific.user_id')}")
    print(f"Non-existent setting: {manager.get_setting('non_existent.key', 'default_for_non_existent')}")

    print("\n--- Setting New Values ---")
    manager.set_setting('general_settings.theme', 'blue_theme')
    manager.set_setting('general_settings.font_size', 12)
    manager.set_setting('new_category.setting_a.nested_b', 'value_for_b')
    manager.set_setting('notifications.sound', 'beep.wav') # Modifying existing category
    print(f"New Theme: {manager.get_setting('general_settings.theme')}")
    print(f"Font Size: {manager.get_setting('general_settings.font_size')}")
    print(f"Nested B: {manager.get_setting('new_category.setting_a.nested_b')}")
    print(f"Notification Sound: {manager.get_setting('notifications.sound')}")


    print("\n--- Current Settings (before save) ---")
    print(json.dumps(manager.settings, indent=4))

    print("\n--- Saving Settings ---")
    manager.save_settings()
    print(f"Settings saved to {manager.config_file_path}")

    print("\n--- Initializing another SettingsManager to verify persistence ---")
    manager2 = SettingsManager('config.json')
    print("\n--- Settings in new manager instance (should reflect saved changes) ---")
    print(json.dumps(manager2.settings, indent=4))
    print(f"Theme from manager2: {manager2.get_setting('general_settings.theme')}")
    print(f"Font Size from manager2: {manager2.get_setting('general_settings.font_size')}")
    print(f"Nested B from manager2: {manager2.get_setting('new_category.setting_a.nested_b')}")

    # Test case: Non-existent config file
    print("\n--- Test with a non-existent config file (should use defaults) ---")
    if os.path.exists('non_existent_config.json'):
        os.remove('non_existent_config.json')
    manager_non_existent = SettingsManager('non_existent_config.json')
    print(json.dumps(manager_non_existent.settings, indent=4))
    print(f"Theme (non-existent file): {manager_non_existent.get_setting('general_settings.theme')}") # Should be 'light'
    manager_non_existent.set_setting('general_settings.new_default_setting', 'test_value')
    manager_non_existent.save_settings() # This will create non_existent_config.json
    print(f"Saved settings to {manager_non_existent.config_file_path}")
    manager_newly_created = SettingsManager('non_existent_config.json')
    print(f"New default setting from newly created file: {manager_newly_created.get_setting('general_settings.new_default_setting')}")

    # Test case: Invalid JSON file
    print("\n--- Test with an invalid JSON file (should use defaults) ---")
    with open('invalid_config.json', 'w') as f:
        f.write("{'invalid_json': True,}") # Invalid JSON (single quotes, trailing comma)
    manager_invalid = SettingsManager('invalid_config.json')
    print(json.dumps(manager_invalid.settings, indent=4))
    print(f"Theme (invalid file): {manager_invalid.get_setting('general_settings.theme')}") # Should be 'light'
    
    # Clean up dummy files
    # os.remove('config.json')
    # os.remove('non_existent_config.json')
    # os.remove('invalid_config.json')
    print("\n--- Example usage finished. Clean up dummy files if desired. ---")
