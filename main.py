import argparse
from train import train
from eval import eval
from generate import generate


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train")
    subparsers.add_parser("eval")
    gen = subparsers.add_parser("generate")
    gen.add_argument("--config", required=True, help="Path to generation config yaml.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"train": train, "eval": eval, "generate": generate}[args.command](args)
