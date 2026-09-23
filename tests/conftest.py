# Makes the modules in workspace/ importable from the tests, since the
# project code imports its siblings by bare module name.
import os
import sys

WORKSPACE = os.path.join(os.path.dirname(__file__), os.pardir, "workspace")
sys.path.insert(0, os.path.abspath(WORKSPACE))
