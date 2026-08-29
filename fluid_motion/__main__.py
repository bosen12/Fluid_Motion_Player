from fluid_motion.app import main

if __name__ == "__main__":
    # SystemExit, not a bare call: main() returns 1 when pywebview is
    # missing, and without this the process still exited 0 (measured).
    # packaging/launch.py already does this, so only the source path
    # was affected.
    raise SystemExit(main())
