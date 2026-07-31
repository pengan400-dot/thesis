#!/usr/bin/env python3
"""Capture the executable training environment for the freeze manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import joblib
    import numpy
    import pandas
    import scipy
    import sklearn
    import torch
    import yaml

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    freeze_path = args.output_dir / "pip_freeze.txt"
    freeze_path.write_text(freeze, encoding="utf-8")
    payload = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pyyaml": yaml.__version__,
            "torch": torch.__version__,
        },
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "gpu_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "pip_freeze_sha256": hashlib.sha256(
            freeze.encode("utf-8")
        ).hexdigest(),
    }
    (args.output_dir / "runtime_environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[PASS] runtime environment: "
        f"torch={torch.__version__}, cuda={torch.cuda.is_available()}"
    )


if __name__ == "__main__":
    main()
