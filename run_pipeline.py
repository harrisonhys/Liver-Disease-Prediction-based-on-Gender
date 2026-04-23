from __future__ import annotations

import argparse

from src.stage2_data_understanding import run_stage_2
from src.stage3_data_preparation import run_stage_3
from src.stage4_modeling import run_stage_4
from src.stage5_evaluation import run_stage_5
from src.stage6_research_packaging import run_stage_6


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the liver disease project pipeline.")
    parser.add_argument(
        "--stage",
        choices=["stage2", "stage3", "stage4", "stage5", "stage6", "all"],
        default="stage6",
        help="Pipeline stage to execute.",
    )
    args = parser.parse_args()

    if args.stage in {"stage2", "all"}:
        run_stage_2()

    if args.stage in {"stage3", "all"}:
        run_stage_3()

    if args.stage in {"stage4", "all"}:
        run_stage_4()

    if args.stage in {"stage5", "all"}:
        run_stage_5()

    if args.stage in {"stage6", "all"}:
        run_stage_6()


if __name__ == "__main__":
    main()