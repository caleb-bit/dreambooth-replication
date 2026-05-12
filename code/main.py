import argparse
from train import train
from eval import eval
from generate import generate


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    trn = subparsers.add_parser("train")
    trn.add_argument("--config", required=True, help="Path to subjects config yaml.")
    trn.add_argument("--subject", required=True, help="Train a single subject by name")
    trn.add_argument("--instance-dir", default="dreambooth_dataset", help="Root dir of per-subject instance image folders.")
    trn.add_argument("--class-dir", default="artifacts/class_images", help="Root dir of per-class regularization image folders.")
    trn.add_argument("--output-dir", default="checkpoints", help="Root dir to write fine-tuned pipeline checkpoints.")
    trn.add_argument("--device", default="cuda")


    evl = subparsers.add_parser("eval")
    evl.add_argument("--config", required=True, help="Path to subjects config yaml.")
    evl.add_argument("--stage", required=True, choices=["generate", "metrics"], help="'generate': produce eval images from fine-tuned checkpoints. 'metrics': compute DINO/CLIP-I/CLIP-T.")
    evl.add_argument("--subject", default=None, help="Evaluate a single subject by name; evaluates all if omitted.")
    evl.add_argument("--checkpoint-dir", default="checkpoints", help="Root dir of fine-tuned pipeline checkpoints (used by generate stage).")
    evl.add_argument("--instance-dir", default="dreambooth_dataset", help="Root dir of per-subject instance images (reference set for DINO/CLIP-I).")
    evl.add_argument("--output-dir", default="artifacts/eval", help="Root dir to write generated eval images into.")
    evl.add_argument("--results-dir", default="results", help="Directory to write metrics.json into.")
    evl.add_argument("--device", default="cuda")


    gen = subparsers.add_parser("generate")
    gen.add_argument("--config", required=True, help="Path to generation config yaml.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"train": train, "eval": eval, "generate": generate}[args.command](args)
