import argparse
from pathlib import Path

from .data import ARTIFACTS, DEFAULT_DATA, DEFAULT_LABELS


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("entier positif attendu")
    return number


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("valeur attendue dans ]0, 1]")
    return number


def classifier_options(parser):
    parser.add_argument("--bundle", type=Path, default=ARTIFACTS / "classifier",
                        help="dossier contenant model.tflite et classes.json")
    parser.add_argument("--detector", type=Path, help="poids .pt du détecteur facultatif")
    parser.add_argument("--detector-size", type=positive, default=960)
    parser.add_argument("--detector-threshold", type=probability, default=0.25)
    parser.add_argument("--device", help="périphérique YOLO, par exemple cpu ou 0")
    parser.add_argument("--min-size", type=positive, default=16, help="côté minimal pour Canny")
    parser.add_argument("--max-size", type=positive, help="côté maximal facultatif pour Canny")
    parser.add_argument("--canny", type=positive, nargs=2, default=(100, 200),
                        metavar=("BAS", "HAUT"))


def get_parser():
    parser = argparse.ArgumentParser(description="Reconnaissance de panneaux routiers")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="entraîner et exporter le classifieur")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--extra-data", type=Path, help="données cibles supplémentaires (train.txt)")
    train.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    train.add_argument("--output", type=Path, default=ARTIFACTS / "classifier")
    train.add_argument("--size", type=positive, default=64)
    train.add_argument("--grayscale", action="store_true")
    train.add_argument("--batch", type=positive, default=64)
    train.add_argument("--epochs", type=positive, default=50)
    train.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser("evaluate", help="évaluer le classifieur ou la chaîne complète")
    classifier_options(evaluate)
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--split", default="val")
    evaluate.add_argument("--output", type=Path, help="rapport JSON facultatif")
    evaluate.add_argument("--batch", type=positive, default=16)
    evaluate.add_argument("--limit", type=positive, help="limiter le nombre d'images pour un essai")
    evaluate.add_argument("--pipeline", action="store_true", help="évaluer aussi la détection")
    evaluate.add_argument("--thresholds", type=probability, nargs="+", default=(0.4, 0.6, 0.8))
    evaluate.add_argument("--iou", type=probability, default=0.5)

    live = commands.add_parser("live", help="reconnaître les panneaux en vidéo")
    classifier_options(live)
    source = live.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--window", help="titre exact d'une fenêtre Windows")
    source.add_argument("--video", type=Path, help="fichier vidéo à lire")
    live.add_argument("--threshold", type=probability, default=0.8)

    prepare = commands.add_parser("prepare-detector", help="exporter les annotations au format YOLO")
    prepare.add_argument("--data", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--extra-data", type=Path)
    prepare.add_argument("--output", type=Path, default=ARTIFACTS / "detector-data")

    detector = commands.add_parser("train-detector", help="entraîner le détecteur YOLO")
    detector.add_argument("--dataset", type=Path, default=ARTIFACTS / "detector-data" / "data.yaml")
    detector.add_argument("--output", type=Path, default=ARTIFACTS / "detector")
    detector.add_argument("--weights", default="yolo11n.pt")
    detector.add_argument("--size", type=positive, default=960)
    detector.add_argument("--epochs", type=positive, default=50)
    detector.add_argument("--device")
    detector.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if hasattr(args, "canny") and args.canny[0] >= args.canny[1]:
        parser.error("le seuil bas de Canny doit être inférieur au seuil haut")
    if hasattr(args, "max_size") and args.max_size and args.max_size < args.min_size:
        parser.error("--max-size doit être supérieur ou égal à --min-size")
    try:
        if args.command == "train":
            from .train import train
            if args.size < 16:
                parser.error("--size doit valoir au moins 16")
            train(args)
        elif args.command == "evaluate":
            from .evaluate import evaluate
            evaluate(args)
        elif args.command == "live":
            from .live import live
            live(args)
        elif args.command == "prepare-detector":
            from .detector import prepare
            prepare(args)
        else:
            from .detector import train
            train(args)
    except ModuleNotFoundError as error:
        if error.name == "ultralytics":
            parser.exit(1, "Installe Ultralytics ou lance cette commande avec uv run --with ultralytics.\n")
        raise
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
