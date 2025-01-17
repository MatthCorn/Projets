#!/bin/bash

# Définir le chemin local du projet
PROJECT_DIR="C:/Users/Matth/Documents/Projets"

# Ajouter le chemin au PYTHONPATH, en évitant un ':' inutile
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PROJECT_DIR"
else
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
fi

# Écrire les paramètres dans un fichier JSON
cat <<EOF > params.json
{
  "n_encoder": 10,
  "len_in": 10,
  "len_out": 5,
  "path_ini": null,
  "d_in": 10,
  "d_att": 128,
  "WidthsEmbedding": [32],
  "n_heads": 4,
  "norm": "post",
  "dropout": 0,
  "lr": 0.0003,
  "mult_grad": 10000,
  "weight_decay": 0.001,
  "NDataT": 500000,
  "NDataV": 1000,
  "batch_size": 1000,
  "n_iter": 80,
  "training_strategy": [
    {"mean": [-50, 50], "std": [0.1, 10]},
    {"mean": [-200, 200], "std": [0.1, 50]}
  ],
  "distrib": "uniform",
  "max_lr": 5,
  "FreqGradObs": 0.333333,
  "warmup": 2
}
EOF

# Appeler le script Python avec le fichier JSON comme argument
python RankAI/V0/Vecteurs/Trainer2.py params.json

# Nettoyer le fichier JSON (facultatif)
rm params.json

echo "Appuyez sur une touche pour fermer la fenêtre."
read -n 1
