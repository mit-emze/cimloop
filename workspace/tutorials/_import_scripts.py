import os
import sys

# fmt: off
THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..')))
import scripts
# fmt: on
