import argparse
import json
from pathlib import Path
import sys

from .policy import GENERAL, PROFILES
from .routing import Gate
from .sparse import SparseModel


def main():
    parser = argparse.ArgumentParser(description="Run the local legal request classifier over JSON.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--profile", default=GENERAL.name, choices=PROFILES)
    args = parser.parse_args()
    try:
        if (args.model / "model.onnx").is_file():
            from .encoder_runtime import EncoderModel

            model = EncoderModel(args.model)
        else:
            path = args.model / "weights.json.gz" if args.model.is_dir() else args.model
            model = SparseModel(path)
        result = Gate(model, PROFILES[args.profile]).decide(json.load(sys.stdin))
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
