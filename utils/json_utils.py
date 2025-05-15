#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON Utilities for handling numpy data types in serialization
"""

import numpy as np

def convert_numpy_types(obj):
    """
    Convert numpy data types to Python native types for JSON serialization
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, or scalar)
        
    Returns:
        object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj
