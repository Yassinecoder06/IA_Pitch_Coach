import os
import sys

# Tests must never touch the real data/secret.key
os.environ.setdefault("SECRET_KEY", "test-only-secret-key")

# Ensure project root is in sys.path for test imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
