from pathlib import Path
import argparse

from analysis import run


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="ЛР5: ковзні методи та прогнозування")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.simulations < 30:
        parser.error("--simulations must be at least 30")
    run(args.data, args.output, args.simulations, args.seed)
