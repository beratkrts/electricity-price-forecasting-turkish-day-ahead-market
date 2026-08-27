import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
"""Database ETL Pipeline Entry Point.

Direct wrapper for running the in-memory EPİAŞ & Weather ETL pipeline.
"""

from daily_update_pipeline import run_daily_pipeline

if __name__ == "__main__":
    run_daily_pipeline()
