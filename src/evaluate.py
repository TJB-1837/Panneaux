import json

import numpy as np

from .data import read_split
from .detector import proposals
from .model import Classifier
from .vision import filter_predictions, intersection_over_union, read_image


def classification_metrics(confusion, names):
    correct = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros(len(names)), where=denominator != 0)

    recall = ratio(correct, support)
    precision = ratio(correct, predicted)
    f1 = ratio(2 * correct, support + predicted)
    return {
        "accuracy": float(correct.sum() / support.sum()),
        "f1_macro": float(f1[support > 0].mean()),
        "per_class": {name: {"support": int(n), "precision": float(p), "recall": float(r), "f1": float(f)}
                      for name, n, p, r, f in zip(names, support, precision, recall, f1)},
        "confusion": confusion.tolist(),
    }


def match_boxes(predictions, truth, min_iou):
    candidates = []
    for p, (box, label) in enumerate(predictions):
        for t, (true_box, true_label) in enumerate(truth):
            if label is None or label == true_label:
                iou = intersection_over_union(box, true_box)
                if iou >= min_iou:
                    candidates.append((iou, p, t))
    found, used = set(), set()
    for _, p, t in sorted(candidates, reverse=True):
        if p not in used and t not in found:
            used.add(p)
            found.add(t)
    return found


def size_group(box):
    x1, y1, x2, y2 = box
    side = min(x2 - x1, y2 - y1)
    if side <= 32:
        return "<=32 px"
    if side <= 50:
        return "33-50 px"
    return ">50 px"


class DetectionMetrics:
    def __init__(self):
        self.predictions = 0
        self.total = dict.fromkeys(("<=32 px", "33-50 px", ">50 px"), 0)
        self.found = self.total.copy()

    def add(self, predictions, truth, min_iou):
        found = match_boxes(predictions, truth, min_iou)
        self.predictions += len(predictions)
        for index, (box, _) in enumerate(truth):
            group = size_group(box)
            self.total[group] += 1
            self.found[group] += index in found

    def report(self, images):
        found, total = sum(self.found.values()), sum(self.total.values())
        return {
            "precision": found / self.predictions if self.predictions else None,
            "recall": found / total if total else None,
            "false_positives_per_image": (self.predictions - found) / images,
            "recall_by_size": {size: {"support": count, "recall": self.found[size] / count if count else None}
                               for size, count in self.total.items()},
        }


def evaluate(args):
    classifier = Classifier(args.bundle, batch_size=args.batch, num_threads=args.threads)
    classes = {name: index for index, name in enumerate(classifier.names)}
    scenes = read_split(args.data, args.split, limit=args.limit)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    if args.pipeline:
        detect = proposals(args)
        raw = DetectionMetrics()
        final = {threshold: DetectionMetrics() for threshold in args.thresholds}

    for scene in scenes:
        truth = [(item.box, classes[item.label]) for item in scene.objects
                 if item.label in classes and not item.ambiguous]
        if not truth and not args.pipeline:
            continue
        frame = read_image(scene.image)
        for (_, predicted, _), (_, actual) in zip(
            classifier.classify(frame, [box for box, _ in truth]), truth
        ):
            confusion[actual, predicted] += 1
        if args.pipeline:
            boxes = detect(frame)
            raw.add([(box, None) for box in boxes], truth, args.iou)
            estimates = list(classifier.classify(frame, boxes))
            for threshold, metric in final.items():
                accepted = filter_predictions(estimates, threshold)
                metric.add([(box, label) for box, label, _ in accepted],
                           truth, args.iou)

    report = {"images": len(scenes), "classes": classifier.names}
    if confusion.sum():
        report["classification"] = classification_metrics(confusion, classifier.names)
    if args.pipeline:
        report["detection_settings"] = {
            "detector": str(args.detector) if args.detector else args.proposal_mode,
            "max_candidates": args.max_candidates,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "canny": list(args.canny),
            "detector_size": args.detector_size,
            "detector_threshold": args.detector_threshold,
        }
        report["iou"] = args.iou
        report["proposals"] = raw.report(len(scenes))
        report["pipeline"] = {str(threshold): metric.report(len(scenes))
                              for threshold, metric in final.items()}
    if not confusion.sum() and not args.pipeline:
        raise ValueError(f"Aucun panneau correspondant aux classes du modèle dans '{args.split}'")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if "classification" in report:
        summary = report["classification"]
        print(f"Classification : accuracy {summary['accuracy']:.1%}, F1 macro {summary['f1_macro']:.1%}")
    if args.pipeline:
        print(json.dumps({key: report[key] for key in ("proposals", "pipeline")}, indent=2))
