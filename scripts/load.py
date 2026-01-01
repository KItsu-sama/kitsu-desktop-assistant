#!/usr/bin/env python3
"""CLI wrapper: scripts.load

Supports:
 - create --out PATH  (write a minimal modelfile to PATH)
 - ollama --dry-run    (no-op dry run)
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="scripts.load")
    subparsers = parser.add_subparsers(dest="cmd")

    p_create = subparsers.add_parser("create")
    p_create.add_argument("--out", required=True, type=Path)

    p_ollama = subparsers.add_parser("ollama")
    p_ollama.add_argument("--dry-run", dest="dry_run", action="store_true")

    args = parser.parse_args()

    if args.cmd == "create":
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        content = "# Modelfile (test)\nFROM /dev/null\n"
        out.write_text(content, encoding="utf-8")
        print(f"created: {out}")
        return 0

    if args.cmd == "ollama":
        if getattr(args, "dry_run", False):
            print("dry-run ok")
            return 0
        # otherwise try delegated implementation
        try:
            from scripts import load_to_ollama_direct
            return load_to_ollama_direct.main()
        except Exception as e:
            print(f"error: {e}")
            return 1

    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())