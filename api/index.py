"""Vercel serverless entry point — exposes the Flask WSGI app."""

import os
import sys

# Make the project root importable so `from app import app` resolves.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import app  # noqa: E402,F401  — Vercel's Python runtime auto-detects `app`
