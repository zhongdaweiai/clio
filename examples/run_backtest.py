"""End-to-end runnable demo. Equivalent to `python -m clio.cli demo`."""

from clio.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo"]))
