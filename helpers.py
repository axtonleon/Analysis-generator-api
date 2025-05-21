# helpers.py
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def deep_convert_numpy_to_python(data):
    """Recursively convert numpy objects (like ndarray) to standard Python types."""
    if isinstance(data, np.ndarray):
        # Convert NumPy array to a Python list
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating, np.bool_, np.datetime64, np.timedelta64)):
         # Convert NumPy scalar types to their Python equivalents
         return data.item() if not pd.isna(data) else None # Use item() for scalars, handle NaT/NaN
    elif isinstance(data, dict):
        # Recurse into dictionary values
        return {k: deep_convert_numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recurse into list items
        return [deep_convert_numpy_to_python(item) for item in data]
    else:
        # Return data as is if it's not a known NumPy type, dict, or list
        return data