import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "Full_Data"
DEFAULT_LABELS = ROOT / "labels401.csv"
ARTIFACTS = ROOT / "artifacts"
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Object:
    box: Box
    label: str
    ambiguous: bool
    occluded: bool


@dataclass(frozen=True)
class Scene:
    image: Path
    width: int
    height: int
    objects: tuple[Object, ...]


def read_labels(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = sorted((int(r["ClassId"]), r["Name"].strip()) for r in csv.DictReader(file))
    names = [name for _, name in rows]
    if ([index for index, _ in rows] != list(range(len(rows)))
            or not names or not all(names) or len(set(names)) != len(names)):
        raise ValueError(f"Identifiants de classe ou noms invalides dans {path}")
    return names


def clip_box(box, width, height):
    left, top, right, bottom = box
    return (max(0, min(width, math.floor(left))), max(0, min(height, math.floor(top))),
            max(0, min(width, math.ceil(right))), max(0, min(height, math.ceil(bottom))))


def load_scene(image, annotation):
    data = json.loads(annotation.read_text(encoding="utf-8"))
    width, height = data["width"], data["height"]
    objects = []
    for item in data["objects"]:
        properties = item.get("properties", {})
        if properties.get("dummy"):
            continue
        bbox = item["bbox"]
        box = clip_box((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]), width, height)
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            objects.append(Object(box, item["label"], bool(properties.get("ambiguous")),
                                  bool(properties.get("occluded"))))
    return Scene(image, width, height, tuple(objects))


def read_split(data_root, split, limit=None):
    annotations_root = data_root
    if not (annotations_root / "annotations").is_dir():
        annotations_root = data_root / "data1" / "mtsd_v2_fully_annotated"
    split_path = annotations_root / "splits" / f"{split}.txt"
    keys = [key.strip() for key in split_path.read_text(encoding="utf-8-sig").splitlines()
            if key.strip()]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(f"Partition vide ou contenant des doublons : {split_path}")
    if any(key in {".", ".."} or "/" in key or "\\" in key for key in keys):
        raise ValueError(f"Clé d'image invalide dans {split_path}")
    if not (annotations_root / "annotations" / f"{keys[0]}.json").is_file():
        raise ValueError(f"Annotations indisponibles pour '{split}' dans {data_root}")
    return [
        load_scene(data_root / "images" / f"{key}.jpg",
                   annotations_root / "annotations" / f"{key}.json")
        for key in keys[:limit]
    ]


def check_disjoint(train, validation):
    overlap = {scene.image.resolve() for scene in train}.intersection(
        scene.image.resolve() for scene in validation
    )
    if overlap:
        raise ValueError(f"{len(overlap)} images communes à l'entraînement et à la validation")
