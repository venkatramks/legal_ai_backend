# backend/api/app.py
# Minimal wrapper so Vercel finds the Flask `app` instance under the api/ path.
# Vercel expects Serverless Functions inside `api/`. This file imports the Flask
# WSGI app object from the backend module so Vercel's Python runtime can expose it.

import os
import sys

# Ensure the parent `backend/` directory is on sys.path so `from app import app` works.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import the Flask app instance named `app` from backend/app.py
# Make sure `backend/app.py` defines `app = Flask(__name__)` at module scope.
from app import app  # noqa: E402,F401
