#!/usr/bin/env python3
"""
Debug launcher for Allentown Behavioral Analysis App.

Run this script for development/testing with verbose output.
"""

import sys
import os

# Add the project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def check_imports():
    """Check that all required imports work."""
    print("Checking imports...")

    modules = [
        ("PyQt6.QtWidgets", "PyQt6"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
    ]

    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  {display_name}: OK")
        except ImportError as e:
            print(f"  {display_name}: FAILED - {e}")
            return False

    # Check local modules
    try:
        from core.data_loader import DataLoader
        print("  core.data_loader: OK")
    except ImportError as e:
        print(f"  core.data_loader: FAILED - {e}")
        return False

    try:
        from core.analysis import BehaviorAnalyzer
        print("  core.analysis: OK")
    except ImportError as e:
        print(f"  core.analysis: FAILED - {e}")
        return False

    try:
        from core.figure_generator import FigureGenerator
        print("  core.figure_generator: OK")
    except ImportError as e:
        print(f"  core.figure_generator: FAILED - {e}")
        return False

    try:
        from core.data_exporter import DataExporter
        print("  core.data_exporter: OK")
    except ImportError as e:
        print(f"  core.data_exporter: FAILED - {e}")
        return False

    print("All imports OK!")
    return True


def main():
    """Main entry point for debug mode."""
    print("=" * 60)
    print("Allentown Behavioral Analysis - Debug Mode")
    print("=" * 60)
    print()

    if not check_imports():
        print("\nImport check failed. Please install missing dependencies.")
        sys.exit(1)

    print()
    print("Starting application...")
    print()

    # Import and run main app
    from main import main as app_main
    app_main()


if __name__ == "__main__":
    main()
