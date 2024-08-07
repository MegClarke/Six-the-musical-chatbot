"""This `conftest.py` file modifies the system path to help resolve the imports of modules in the test scripts."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
