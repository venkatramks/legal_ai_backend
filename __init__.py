"""Backend package initializer.

This file makes the `backend` directory a Python package so modules
can be imported as `import backend.app` from the workspace root.
"""

__all__ = ["app", "ocr_service", "llm_service", "supabase_client", "text_cleaner"]
