# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2026 Noxfort Systems
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# File: ui/styles/theme_manager.py
# Author: Gabriel Moraes
# Date: 2026-03-01

import os
import json
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

class ThemeManager:
    """
    Singleton class responsible for loading and providing UI theme properties 
    (colors, fonts, sizes, and raw CSS styles) from a centralized JSON configuration file.
    """
    _instance = None
    _theme_data = {}
    _current_theme = "dark"  # default active profile

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ThemeManager, cls).__new__(cls)
            cls._instance._load_theme()
        return cls._instance

    @classmethod
    def set_theme(cls, theme_name: str):
        """Sets the active system UI theme ('light' or 'dark')."""
        instance = cls()
        if theme_name in ("light", "dark"):
            instance._current_theme = theme_name

    @classmethod
    def apply_theme(cls, theme_index: int):
        """
        Applies a full theme switch across the entire application.
        
        Encapsulates QPalette creation, Fusion style, CSS re-parse, 
        and component-level update_theme() calls.
        
        Args:
            theme_index: 0 = Dark, 1 = Light, 2 = Dark (system default).
        """
        theme_map = {0: "dark", 1: "light", 2: "dark"}
        theme_name = theme_map.get(theme_index, "dark")
        cls.set_theme(theme_name)
        
        app = QApplication.instance()
        if not app:
            return
        
        # Apply native Fusion style with themed QPalette
        app.setStyle("Fusion")
        palette = QPalette()
        
        palette.setColor(QPalette.ColorRole.Window, cls.get_color("card_background"))
        palette.setColor(QPalette.ColorRole.WindowText, cls.get_color("text_main"))
        palette.setColor(QPalette.ColorRole.Base, cls.get_color("background_light"))
        palette.setColor(QPalette.ColorRole.AlternateBase, cls.get_color("card_background"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, cls.get_color("card_background"))
        palette.setColor(QPalette.ColorRole.ToolTipText, cls.get_color("text_main"))
        palette.setColor(QPalette.ColorRole.Text, cls.get_color("text_main"))
        palette.setColor(QPalette.ColorRole.Button, cls.get_color("card_border"))
        palette.setColor(QPalette.ColorRole.ButtonText, cls.get_color("text_main"))
        palette.setColor(QPalette.ColorRole.BrightText, cls.get_color("danger"))
        palette.setColor(QPalette.ColorRole.Link, cls.get_color("primary"))
        palette.setColor(QPalette.ColorRole.Highlight, cls.get_color("primary"))
        
        # Contrast for highlighted text
        hl_text = cls.get_color("text_dark") if theme_name == "light" else cls.get_color("background_light")
        palette.setColor(QPalette.ColorRole.HighlightedText, hl_text)
        
        # Disabled Roles
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, cls.get_color("text_muted"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, cls.get_color("text_muted"))

        app.setPalette(palette)
        
        # Force re-evaluation of baked Python string HEX styles on components
        for widget in app.allWidgets():
            if hasattr(widget, 'update_theme') and callable(widget.update_theme):
                try:
                    widget.update_theme()
                except Exception:
                    pass
        
        # Force native Qt CSS engine to re-parse 'palette()' dynamic bindings
        for widget in app.allWidgets():
            if widget.styleSheet():
                widget.setStyleSheet(widget.styleSheet())

    def _load_theme(self):
        """Loads the theme JSON file. Uses fallback empty dicts if it fails."""
        # Calculate absolute path relative to this file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        theme_path = os.path.join(current_dir, "theme.json")

        try:
            with open(theme_path, 'r', encoding='utf-8') as f:
                self._theme_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load theme.json at {theme_path}. Error: {e}")
            self._theme_data = {
                "themes": {
                    "light": {}, "dark": {}
                },
                "fonts": {},
                "sizes": {},
                "styles": {}
            }

    @classmethod
    def get_color(cls, color_name: str, fallback: str = "#000000") -> QColor:
        """
        Retrieves a QColor object based on the requested theme color name.
        """
        instance = cls()
        theme_profile = instance._theme_data.get("themes", {}).get(instance._current_theme, {})
        hex_code = theme_profile.get(color_name, fallback)
        return QColor(hex_code)

    @classmethod
    def get_hex(cls, color_name: str, fallback: str = "#000000") -> str:
        """
        Retrieves a raw HEX color string, useful for setStyleSheet injections.
        """
        instance = cls()
        theme_profile = instance._theme_data.get("themes", {}).get(instance._current_theme, {})
        return theme_profile.get(color_name, fallback)

    @classmethod
    def get_font(cls, font_type: str = "body_size") -> QFont:
        """
        Retrieves a QFont object with the family and size defined in the theme.
        font_type can be 'title_size', 'subtitle_size', or 'body_size'.
        """
        instance = cls()
        fonts_config = instance._theme_data.get("fonts", {})
        
        family = fonts_config.get("family", "Arial")
        size = fonts_config.get(font_type, 10)
        
        return QFont(family, size)

    @classmethod
    def get_size(cls, size_name: str, fallback=0):
        """
        Retrieves a numeric or string size value (e.g., border radius, padding).
        """
        instance = cls()
        return instance._theme_data.get("sizes", {}).get(size_name, fallback)

    @classmethod
    def get_style(cls, style_name: str, fallback: str = "", **kwargs) -> str:
        """
        Retrieves a full CSS template string from the theme and injects dynamic variables.
        Automatically unpacks all colors from the currently active theme into the kwargs so you 
        don't need to manually fetch hex variables for CSS blocks.
        
        Args:
            style_name (str): The name of the style block in theme.json.
            fallback (str): The string to return if the style is not found.
            **kwargs: Extra variables to inject into the CSS template (e.g., color="#FF0000").
            
        Returns:
            str: The formatted CSS stylesheet string.
        """
        instance = cls()
        style_template = instance._theme_data.get("styles", {}).get(style_name, fallback)
        
        # Load all default colors for the current theme
        theme_profile = instance._theme_data.get("themes", {}).get(instance._current_theme, {})
        
        # Merge manual kwargs directly, overriding defaults if a collision happens
        format_args = {**theme_profile, **kwargs}
        
        if style_template:
            try:
                # Injects the provided variables into the string placeholders {var}
                return style_template.format(**format_args)
            except KeyError as e:
                print(f"⚠️ Warning: Missing CSS variable {e} for style '{style_name}'.")
                return style_template
                
        return style_template