"""
Setup module for Jupyter notebooks
Adds project root to sys.path so imports work from any notebook location
"""

import sys
import os

# Add project root to path - allows importing from root level

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
