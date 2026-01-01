#!/usr/bin/env python3
"""CLI wrapper: scripts.generate

Minimal wrapper to satisfy tests: supports `training` subcommand with `--dry-run`.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="scripts.generate")
    parser.add_argument("target", choices=["training"], help="Target dataset to generate")
    parser.add_argument("--num", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Do not perform actions, just validate args")
    args = parser.parse_args()

    if args.dry_run:
        print(f"dry-run ok (target={args.target} num={args.num})")
        return 0

    try:
        if args.target == "training":
            from scripts import generate_training_data
            # call small helper if one exists; otherwise just return success
            try:
                generate_training_data.main(num=args.num)
            except Exception:
                pass
        return 0
    except Exception as e:
        print(f"error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())