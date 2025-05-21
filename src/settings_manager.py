import json
import os

class SettingsManager:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.settings = {}
        self.default_settings = {
            # TODO: Populate with default settings from main_ui.py and *_LAYOUT_DATA files
            "language": "en",
            "theme": "default",
            "output_directory": "./output",
            "processing_device": "cpu",
            "num_threads": 4,
            "vram_limit_gb": 4,
            "cache_format": "png",
            "cache_quality": 90,
            "log_level": "INFO",
        }
        self.load_settings()

    def load_settings(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                try:
                    loaded_settings = json.load(f)
                    self.settings = {**self.default_settings, **loaded_settings}
                except json.JSONDecodeError:
                    # Handle malformed JSON, use defaults
                    self.settings = self.default_settings.copy()
        else:
            self.settings = self.default_settings.copy()

    def save_settings(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=4)

    def get_setting(self, key):
        return self.settings.get(key)

    def set_setting(self, key, value):
        self.settings[key] = value

    def get_all_settings(self):
        return self.settings.copy()

if __name__ == '__main__':
    # Example Usage
    settings_manager = SettingsManager()
    print("Initial settings:", settings_manager.get_all_settings())
    settings_manager.set_setting("theme", "dark")
    settings_manager.save_settings()
    print("Settings after change:", settings_manager.get_all_settings())

    # Test loading from existing file
    new_settings_manager = SettingsManager()
    print("Settings loaded from file:", new_settings_manager.get_all_settings())
    # Clean up the created config file
    if os.path.exists(settings_manager.config_file):
        os.remove(settings_manager.config_file)
        print(f"Cleaned up {settings_manager.config_file}")
