"""Core configuration for the Essential Open API application."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directory that stores Java (.jar) dependencies used via JPype.
JARS_DIR = BASE_DIR / "jars"

# Directory for Protégé project files (.pprj).
PPRJ_DIR = BASE_DIR / "resources"
PPRJ_FILENAME = "essential_baseline_6_20.pprj"
PPRJ_FILE = PPRJ_DIR / PPRJ_FILENAME
