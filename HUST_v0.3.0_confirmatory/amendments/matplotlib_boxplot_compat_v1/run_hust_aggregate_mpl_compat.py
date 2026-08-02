from __future__ import annotations

import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from hust_v030.evaluation import aggregate_confirmation


AMENDMENT_ID = "HUST_MATPLOTLIB_BOXPLOT_TICK_LABELS_v1"

original_boxplot = Axes.boxplot
parameters = inspect.signature(original_boxplot).parameters

if "labels" not in parameters and "tick_labels" in parameters:

    def boxplot_compat(self, *args, **kwargs):
        if "labels" in kwargs:
            if "tick_labels" in kwargs:
                raise TypeError(
                    "Both labels and tick_labels were supplied"
                )

            kwargs["tick_labels"] = kwargs.pop("labels")

        return original_boxplot(self, *args, **kwargs)

    Axes.boxplot = boxplot_compat

elif "labels" not in parameters:
    raise RuntimeError(
        "Axes.boxplot accepts neither labels nor tick_labels"
    )


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: run_hust_aggregate_mpl_compat.py "
            "CONFIG CONFIRMATION OUTPUT"
        )

    print(
        f"[AMENDMENT] {AMENDMENT_ID} "
        f"matplotlib={matplotlib.__version__}"
    )

    aggregate_confirmation(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
    )


if __name__ == "__main__":
    main()
