"""Command-line entry point: ``python -m beacon --config <yaml>``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import ConfigError, load_config
from .runner import Runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beacon",
        description="Run the BEACON CTI knowledge-graph pipeline from a YAML configuration.",
    )
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML configuration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate inputs and print the plan without running anything")
    parser.add_argument("--version", action="version", version=f"beacon {__version__}")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        runner = Runner(cfg, dry_run=args.dry_run)
        return runner.run()
    except ConfigError as exc:
        print(f"[beacon] configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[beacon] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
