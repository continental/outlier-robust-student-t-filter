# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Loads the src modules so that other scripts can use them
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#print("Sys path:  ", sys.path)
import src.filters as filters
import src.visual as visual
import src.eval as eval
import src.utils as utils