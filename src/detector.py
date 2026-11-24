import json
import os
import shutil
from pathlib import Path

from .data import check_disjoint, clip_box, read_split
from .vision import canny_boxes, color_boxes, suppress_overlaps


def proposals(args):
    if args.detector is None:
        def detect(frame):
            if args.proposal_mode == "color":
                boxes = color_boxes(frame, args.min_size, args.max_size)
            else:
                boxes = list(canny_boxes(frame, args.min_size, args.max_size, *args.canny))
                boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            return suppress_overlaps(boxes, limit=args.max_candidates)
        return detect

    from ultralytics import YOLO

    model = YOLO(str(args.detector), task="detect")

    def detect(frame):
        height, width = frame.shape[:2]
        result, = model.predict(frame, imgsz=args.detector_size, conf=args.detector_threshold,
                                device=args.device, max_det=args.max_candidates, verbose=False)
        boxes = [clip_box(box, width, height) for box in result.boxes.xyxy.cpu().tolist()]
        return [box for box in boxes if box[2] > box[0] and box[3] > box[1]]

    return detect


def link_image(source, target):
    # Ultralytics can repair JPEGs that lack an end marker in place.
    with source.open("rb") as image:
        image.seek(-2, os.SEEK_END)
        complete = image.read() == b"\xff\xd9"
    if complete:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copyfile(source, target)


def write_split(output, split, scene_groups):
    images = output / "images" / split
    labels = output / "labels" / split
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    box_count = 0
    for source_id, scenes in enumerate(scene_groups):
        for scene in scenes:
            base = f"{source_id}_{scene.image.stem}"
            link_image(scene.image, images / f"{base}.jpg")
            rows = []
            for item in scene.objects:
                x1, y1, x2, y2 = item.box
                cx = (x1 + x2) / (2 * scene.width)
                cy = (y1 + y2) / (2 * scene.height)
                w, h = (x2 - x1) / scene.width, (y2 - y1) / scene.height
                rows.append(f"0 {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")
            box_count += len(rows)
            (labels / f"{base}.txt").write_text("\n".join(rows), encoding="utf-8")
    if box_count == 0:
        raise ValueError(f"Aucun panneau annoté dans la partition '{split}'")
    print(f"{split} : {box_count} boîtes pour le détecteur", flush=True)


def prepare(args):
    train_scenes = read_split(args.data, "train")
    validation = read_split(args.data, "val")
    check_disjoint(train_scenes, validation)
    train_groups = [train_scenes]
    if args.extra_data:
        extra = read_split(args.extra_data, "train")
        check_disjoint(extra, train_scenes + validation)
        train_groups.append(extra)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_split(output, "train", train_groups)
    write_split(output, "val", [validation])
    dataset = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "panneau"},
    }
    (output / "data.yaml").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Données de détection prêtes dans {output}")


def train(args):
    if not args.dataset.is_file():
        raise ValueError(f"Données YOLO introuvables : {args.dataset}. Lance prepare-detector.")

    from ultralytics import YOLO

    model = YOLO(args.weights, task="detect")
    model.train(data=str(args.dataset.resolve()), imgsz=args.size, epochs=args.epochs,
                device=args.device, project=str(args.output.resolve()), name="train",
                fliplr=0.0, seed=args.seed)
    print(f"Poids du détecteur : {Path(model.trainer.best)}")
