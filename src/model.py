import json
from pathlib import Path

import numpy as np

from .vision import crop_input


def export_bundle(model, names, destination):
    import tensorflow as tf

    model.save(destination / "model.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    (destination / "model.tflite").write_bytes(converter.convert())
    table = destination / "classes.json.tmp"
    table.write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table.replace(destination / "classes.json")


class Classifier:
    def __init__(self, bundle: Path, batch_size=1, num_threads=4):
        import tensorflow as tf

        self.names = json.loads((bundle / "classes.json").read_text(encoding="utf-8"))
        if (not isinstance(self.names, list) or not self.names
                or not all(isinstance(name, str) and name for name in self.names)
                or len(set(self.names)) != len(self.names)):
            raise ValueError("Table de classes invalide")

        self.interpreter = tf.lite.Interpreter(
            model_path=str(bundle / "model.tflite"), num_threads=num_threads
        )
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1 or len(inputs[0]["shape"]) != 4:
            raise ValueError("Signature du classifieur TFLite invalide")

        self.input_index = inputs[0]["index"]
        self.output_index = outputs[0]["index"]
        batch, self.height, self.width, self.channels = map(int, inputs[0]["shape"])
        if (self.channels not in (1, 3) or min(self.height, self.width, batch) < 1
                or inputs[0]["dtype"] != np.float32 or outputs[0]["dtype"] != np.float32
                or tuple(outputs[0]["shape"]) != (batch, len(self.names))):
            raise ValueError("Dimensions, types ou nombre de classes incompatibles avec le modèle")

        if inputs[0]["shape_signature"][0] == -1 and batch != batch_size:
            batch = batch_size
            self.interpreter.resize_tensor_input(
                self.input_index, [batch, self.height, self.width, self.channels], strict=True
            )
        self.interpreter.allocate_tensors()
        self.batch = np.zeros((batch, self.height, self.width, self.channels), dtype=np.float32)

    def classify(self, frame, boxes):
        size = len(self.batch)
        for start in range(0, len(boxes), size):
            chunk = boxes[start:start + size]
            for i, box in enumerate(chunk):
                self.batch[i] = crop_input(frame, box, self.width, self.height, self.channels)
            self.interpreter.set_tensor(self.input_index, self.batch)
            self.interpreter.invoke()
            scores = self.interpreter.get_tensor(self.output_index)[:len(chunk)]
            winners = scores.argmax(axis=1)
            yield from ((box, int(label), float(row[label]))
                        for box, row, label in zip(chunk, scores, winners))
