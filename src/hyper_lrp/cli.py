from __future__ import annotations

import argparse
from pathlib import Path

from hyper_lrp.service import LRPService


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LRP explanations for supported adapters.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available adapters")
    list_parser.set_defaults(command="list")

    explain_parser = subparsers.add_parser("explain", help="Explain an image with an adapter")
    explain_parser.add_argument("--adapter", required=True, help="Adapter name, e.g. hf-clip")
    explain_parser.add_argument("--image", required=True, type=Path, help="Path to the input image")
    explain_parser.add_argument("--prompt", required=True, help="Target prompt to explain")
    explain_parser.add_argument("--device", default=None, help="Torch device override")
    explain_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./artifacts/latest"),
        help="Where to write explanation artifacts",
    )
    explain_parser.set_defaults(command="explain")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    service = LRPService()

    if args.command == "list":
        for adapter_name in service.list_adapters():
            print(adapter_name)
        return

    result = service.explain_path(
        adapter_name=args.adapter,
        image_path=args.image,
        prompt=args.prompt,
        device=args.device,
    )
    saved = result.save(args.output_dir)
    print(f"adapter={result.adapter_name}")
    print(f"prompt={result.prompt}")
    print(f"score={result.score:.6f}")
    for key, path in saved.items():
        print(f"{key}={path}")