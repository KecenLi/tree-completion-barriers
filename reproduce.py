"""Reproduce the construction tables and run the associated regression tests."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    (ROOT / "outputs").mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for name in ["tree_planner", "noisy_tree", "operator_sensitivity", "cycle_scope"]:
        subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py")], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "--rootdir=.", "tests"], cwd=ROOT, env=env, check=True)

if __name__ == "__main__":
    main()
