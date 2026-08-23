from __future__ import annotations

import multiprocessing
import sys


if __name__ == "__main__":
    multiprocessing.freeze_support()
    from fluid_motion.app import main

    raise SystemExit(main())
