# Reconnaissance de panneaux routiers

La version active est dans `src`. Les scripts, les modèles et les tables de classes de la version précédente sont conservés dans `Old_files/V3`. `labels401.csv` à la racine est la table de référence des 89 classes pour le nouvel entraînement.

## Entraîner et lancer la reconnaissance

Depuis la racine, avec Python 3.13 et `uv` :

```powershell
uv sync
uv run python -m src train
uv run python -m src live --camera 0
uv run python -m src live --window 'Euro Truck Simulator 2'
uv run python -m src live --video captures/trajet.mp4
```

L’entraînement utilise les partitions `train.txt` et `val.txt` de Mapillary. Il enregistre les modèles Keras et TFLite, la table de classes associée et l’historique dans `artifacts/classifier`. La version par défaut utilise des images couleur de 64 × 64, avec conservation des proportions et augmentation à l’entraînement. Pour comparer avec les niveaux de gris, entraîne une deuxième version avec `--grayscale --output artifacts/classifier-gray`, puis utilise `--bundle artifacts/classifier-gray` pour l’évaluer. Chaque entraînement demande un nouveau dossier de sortie. Pour la capture d’écran, garde la fenêtre cible ouverte et utilise le mode fenêtré si le plein écran ne peut pas être capturé.

## Évaluer

```powershell
uv run python -m src evaluate --output artifacts/classification.json
uv run python -m src evaluate --pipeline --output artifacts/canny.json
uv run python -m src evaluate --pipeline --min-size 51 --max-size 299 --output artifacts/canny-filtre-v3.json
```

La première commande mesure le F1 macro, le rappel par classe et la matrice de confusion sur les boîtes annotées. Avec `--pipeline`, le rapport ajoute le rappel des propositions par taille et le rappel et les faux positifs par image après classification. Les seuils sont réglables via `--thresholds` et le seuil retenu s’applique en vidéo avec `--threshold`. Ajoute `--limit 100` pour un essai rapide sur les 100 premières images de la partition.

## Essayer le détecteur entraîné

```powershell
uv run python -m src prepare-detector
uv run --with ultralytics python -m src train-detector
uv run --with ultralytics python -m src evaluate --pipeline --detector artifacts/detector/train/weights/best.pt --output artifacts/yolo.json
uv run --with ultralytics python -m src live --window 'Euro Truck Simulator 2' --detector artifacts/detector/train/weights/best.pt
```

Ultralytics est installé à la demande pour ces commandes. L’entraînement télécharge les poids initiaux de YOLO11n si nécessaire. `prepare-detector` utilise les boîtes Mapillary et crée des liens physiques pour les JPEG compatibles. Une copie est utilisée si le lien ne peut pas être créé. Sur CPU, l’entraînement complet peut prendre du temps. Les options `--epochs`, `--size` et `--device` permettent d’ajuster l’essai.

## Utiliser des captures annotées

Place les JPEG dans `images/`, les JSON au format Mapillary dans `annotations/` et les listes de clés d’image sans extension dans `splits/`. Utilise des trajets distincts pour les partitions et des annotations avec `objects: []` pour les images sans panneau. Les noms de classe doivent correspondre à `labels401.csv`.

- Ajoute `--extra-data captures/annotees` à `train` ou `prepare-detector` pour inclure `captures/annotees/splits/train.txt`.
- Utilise `evaluate --data captures/annotees --split test --pipeline` pour mesurer les résultats sur les trajets réservés. Les annotations de la partition officielle `test.txt` de Mapillary sont absentes de la copie locale.

Les options de chaque commande sont disponibles avec `uv run python -m src <commande> --help`.
