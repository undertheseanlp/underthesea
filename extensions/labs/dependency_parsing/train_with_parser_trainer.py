"""
Example: Training a dependency parser using the simplified ParserTrainer API.

This script demonstrates how to train a dependency parser using the
user-friendly ParserTrainer class introduced in GH-392.

Usage:
    python train_with_parser_trainer.py
"""
from os.path import join

from underthesea.datasets.vlsp2020_dp import VLSP2020_DP
from underthesea.file_utils import MODELS_FOLDER
from underthesea.trainers import ParserTrainer

# Load the VLSP2020 dependency parsing corpus
corpus = VLSP2020_DP()

# Create a trainer with character-level embeddings (default)
trainer = ParserTrainer(corpus, feat='char')

# Train the model
trainer.train(
    output_dir=join(MODELS_FOLDER, 'parsers', 'vi-dp-char'),
    max_epochs=100,
    batch_size=5000,
    lr=2e-3,
    patience=100
)

# Alternative: Train with BERT embeddings
# trainer_bert = ParserTrainer(corpus, feat='bert', bert='vinai/phobert-base')
# trainer_bert.train(
#     output_dir=join(MODELS_FOLDER, 'parsers', 'vi-dp-bert'),
#     max_epochs=100
# )
