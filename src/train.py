import numpy as np

from .data import check_disjoint, read_labels, read_split
from .model import export_bundle
from .vision import crop_input, read_image


def collect(scenes, classes, size, channels, split):
    selected = [
        (scene, [(item.box, classes[item.label]) for item in scene.objects
                 if not item.ambiguous and item.label in classes])
        for scene in scenes
    ]
    count = sum(len(objects) for _, objects in selected)
    if not count:
        raise ValueError(f"Aucun panneau exploitable dans '{split}'")
    print(f"{split} : {count} panneaux dans {sum(bool(items) for _, items in selected)} images",
          flush=True)
    images = np.empty((count, size, size, channels), dtype=np.uint8)
    labels = np.empty(count, dtype=np.int32)
    index = 0
    for scene, objects in selected:
        if not objects:
            continue
        frame = read_image(scene.image)
        for box, label in objects:
            images[index] = crop_input(frame, box, size, size, channels)
            labels[index] = label
            index += 1
    return images, labels


def build_model(tf, size, channels, class_count):
    layers = tf.keras.layers
    inputs = tf.keras.Input((size, size, channels), dtype="float32", name="pixels")
    x = layers.Rescaling(1 / 255)(inputs)
    x = layers.RandomRotation(0.035, fill_mode="nearest")(x)
    x = layers.RandomTranslation(0.08, 0.08, fill_mode="nearest")(x)
    x = layers.RandomZoom(0.12, fill_mode="nearest")(x)
    x = layers.RandomBrightness(0.18, value_range=(0.0, 1.0))(x)
    x = layers.RandomContrast(0.2, value_range=(0.0, 1.0))(x)
    for filters in (32, 64, 128):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(layers.Dense(128, activation="relu")(x))
    model = tf.keras.Model(inputs, layers.Dense(class_count, activation="softmax")(x))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def train(args):
    import tensorflow as tf

    if args.output.exists():
        raise ValueError(f"Le dossier de sortie existe déjà : {args.output}")
    tf.keras.utils.set_random_seed(args.seed)
    names = read_labels(args.labels)
    classes = {name: index for index, name in enumerate(names)}
    train_scenes = read_split(args.data, "train")
    if args.extra_data:
        check_disjoint(train_scenes, extra := read_split(args.extra_data, "train"))
        train_scenes.extend(extra)
    validation_scenes = read_split(args.data, "val")
    check_disjoint(train_scenes, validation_scenes)
    channels = 1 if args.grayscale else 3
    train_images, train_labels = collect(train_scenes, classes, args.size, channels, "train")
    counts = np.bincount(train_labels, minlength=len(names))
    if np.any(counts == 0):
        missing = ", ".join(names[i] for i in np.flatnonzero(counts == 0))
        raise ValueError(f"Classes absentes de l'entraînement : {missing}")
    val_images, val_labels = collect(validation_scenes, classes, args.size, channels, "val")

    weights = (len(train_labels) / (len(names) * counts)).astype(np.float32)
    train_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels, weights[train_labels]))
    validation = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    del train_images, val_images

    random = tf.random.Generator.from_seed(args.seed)

    def motion_blur(images, labels, weights):
        pixels = tf.cast(images, tf.float32)
        horizontal = tf.nn.avg_pool2d(pixels, [1, 3], strides=1, padding="SAME")
        vertical = tf.nn.avg_pool2d(pixels, [3, 1], strides=1, padding="SAME")
        choice = random.uniform((tf.shape(pixels)[0], 1, 1, 1))
        return tf.where(choice < 0.1, horizontal, tf.where(choice < 0.2, vertical, pixels)), labels, weights

    train_set = (train_set.shuffle(min(len(train_labels), 4096), seed=args.seed)
                 .batch(args.batch).map(motion_blur).prefetch(tf.data.AUTOTUNE))
    validation = validation.batch(args.batch).prefetch(tf.data.AUTOTUNE)
    model = build_model(tf, args.size, channels, len(names))
    args.output.mkdir(parents=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
        tf.keras.callbacks.CSVLogger(str(args.output / "training.csv")),
    ]
    model.fit(train_set, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    export_bundle(model, names, args.output)
    print(f"Modèles Keras et TFLite exportés dans {args.output}")
