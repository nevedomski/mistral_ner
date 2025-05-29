#!/usr/bin/env python3
"""Sync offline WandB runs to the cloud."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import list_offline_runs, sync_all_offline_runs, sync_offline_run


def setup_logging() -> logging.Logger:
    """Setup basic logging for the sync script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def main() -> None:
    """Main function for syncing offline WandB runs."""
    parser = argparse.ArgumentParser(description="Sync offline WandB runs to the cloud")
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="./wandb",
        help="Directory containing offline WandB runs (default: ./wandb)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all offline runs without syncing"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Sync specific run by ID (e.g., offline-run-20231120_123456-abc123)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without actually syncing"
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Check if wandb directory exists
    wandb_path = Path(args.wandb_dir)
    if not wandb_path.exists():
        logger.error(f"WandB directory {args.wandb_dir} does not exist")
        sys.exit(1)

    # List offline runs
    offline_runs = list_offline_runs(args.wandb_dir)
    
    if not offline_runs:
        logger.info("No offline runs found")
        return

    if args.list:
        logger.info(f"Found {len(offline_runs)} offline runs:")
        for run in offline_runs:
            logger.info(f"  - {run['run_id']} ({run['size_mb']:.1f} MB)")
        return

    if args.run_id:
        # Sync specific run
        run_path = None
        for run in offline_runs:
            if run["run_id"] == args.run_id:
                run_path = run["path"]
                break
        
        if not run_path:
            logger.error(f"Run {args.run_id} not found")
            sys.exit(1)
        
        if args.dry_run:
            logger.info(f"Would sync run: {args.run_id}")
            return
        
        logger.info(f"Syncing run: {args.run_id}")
        if sync_offline_run(run_path):
            logger.info("✅ Sync completed successfully")
        else:
            logger.error("❌ Sync failed")
            sys.exit(1)
    else:
        # Sync all runs
        if args.dry_run:
            logger.info(f"Would sync {len(offline_runs)} offline runs:")
            for run in offline_runs:
                logger.info(f"  - {run['run_id']}")
            return
        
        logger.info(f"Syncing {len(offline_runs)} offline runs...")
        results = sync_all_offline_runs(args.wandb_dir)
        
        if results["synced"]:
            logger.info(f"✅ Successfully synced {len(results['synced'])} runs:")
            for run_id in results["synced"]:
                logger.info(f"  - {run_id}")
        
        if results["failed"]:
            logger.error(f"❌ Failed to sync {len(results['failed'])} runs:")
            for run_id in results["failed"]:
                logger.error(f"  - {run_id}")
            sys.exit(1)
        
        if not results["synced"] and not results["failed"]:
            logger.info("No runs to sync")


if __name__ == "__main__":
    main()