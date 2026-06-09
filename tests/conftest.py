"""Pytest configuration — stub unavailable optional dependencies.

psycopg2 and google-cloud-storage are runtime dependencies of arpi.analyzer.legacy
that are not installed in the test environment. Stubbing them here allows the
analyzer.legacy modules to be imported during test collection without error.
"""

import sys
from unittest.mock import MagicMock

for _mod in (
    "psycopg2",
    "psycopg2.extras",
    "google.cloud",
    "google.cloud.storage",
):
    sys.modules.setdefault(_mod, MagicMock())
