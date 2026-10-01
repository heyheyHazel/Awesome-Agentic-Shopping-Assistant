"""Test isolation: pin the data source before any app module is imported.

`data/store.py` resolves the catalog at import time. Tests assert on the
built-in mock catalog (stable product IDs, fixed RFM values), so it must be
selected regardless of whether `data/generated/` exists locally.
"""

import os

os.environ["ECOM_DATA_SOURCE"] = "mock"
