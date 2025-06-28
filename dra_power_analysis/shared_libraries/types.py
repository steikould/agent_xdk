"""Common data schema and types for power-optimization agents."""

from typing import Optional, Union

from google.genai import types
from pydantic import BaseModel, Field


# Convenient declaration for controlled generation.
json_response_config = types.GenerateContentConfig(
    response_mime_type="application/json"
)
