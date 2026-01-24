"""
ParserTrainer - A user-friendly trainer for dependency parsing.

This module provides a simplified interface for training dependency parsers
using the Biaffine architecture.

Example usage:
    from underthesea.trainers import ParserTrainer
    from underthesea.datasets.vlsp2020_dp import VLSP2020_DP

    corpus = VLSP2020_DP()
    trainer = ParserTrainer(corpus)
    trainer.train(output_dir='./models/my_parser')
"""
from pathlib import Path
from typing import Union

from underthesea.models.dependency_parser import DependencyParser
from underthesea.modules.embeddings import CharacterEmbeddings, FieldEmbeddings
from underthesea.trainers.dependency_parser_trainer import DependencyParserTrainer


class ParserTrainer:
    """
    A trainer for dependency parsing models.

    This class provides a simplified interface for training dependency parsers
    using the Biaffine architecture with support for character-level and
    BERT-based embeddings.

    Args:
        corpus: A corpus object that provides train/dev/test file paths.
            Must have `train`, `dev`, and `test` attributes pointing to
            CoNLL-format files.
        feat (str): Feature type for the parser. Options:
            - 'char': Character-level LSTM embeddings (default)
            - 'bert': Pre-trained BERT embeddings
            - 'tag': POS tag embeddings
        bert (str, optional): BERT model name/path when feat='bert'.
            Default: 'vinai/phobert-base' for Vietnamese.
        embed (str, optional): Path to pre-trained word embeddings file.

    Example:
        >>> from underthesea.trainers import ParserTrainer
        >>> from underthesea.datasets.vlsp2020_dp import VLSP2020_DP
        >>> corpus = VLSP2020_DP()
        >>> trainer = ParserTrainer(corpus)
        >>> trainer.train(output_dir='./models/my_parser', max_epochs=100)
    """

    def __init__(
        self,
        corpus,
        feat: str = 'char',
        bert: str | None = None,
        embed: str | None = None
    ):
        self.corpus = corpus
        self.feat = feat
        self.bert = bert
        self.embed = embed

        # Set default BERT model for Vietnamese if feat='bert' and no bert specified
        if feat == 'bert' and bert is None:
            self.bert = 'vinai/phobert-base'

        # Build embeddings based on feature type
        self.embeddings = self._build_embeddings()

        # Initialize parser with pre-training flag
        self.parser = DependencyParser(
            embeddings=self.embeddings,
            feat=self.feat,
            bert=self.bert,
            embed=self.embed,
            init_pre_train=True
        )

        # Create the internal trainer
        self._trainer = DependencyParserTrainer(self.parser, self.corpus)

    def _build_embeddings(self) -> list:
        """Build embeddings list based on feature type."""
        embeddings = [FieldEmbeddings()]

        if self.feat == 'char':
            embeddings.append(CharacterEmbeddings())
        # For 'bert' and 'tag', FieldEmbeddings is sufficient
        # as the feat_embed is handled by DependencyParser

        return embeddings

    def train(
        self,
        output_dir: Union[Path, str],
        max_epochs: int = 100,
        batch_size: int = 5000,
        lr: float = 2e-3,
        patience: int = 100,
        fix_len: int = 20,
        min_freq: int = 2,
        buckets: int = 1000,
        mu: float = 0.9,
        nu: float = 0.9,
        epsilon: float = 1e-12,
        clip: float = 5.0,
        decay: float = 0.75,
        decay_steps: int = 5000,
        wandb=None
    ):
        """
        Train the dependency parser.

        Args:
            output_dir (str or Path): Directory to save the trained model.
            max_epochs (int): Maximum number of training epochs. Default: 100.
            batch_size (int): Number of tokens per batch. Default: 5000.
            lr (float): Learning rate for Adam optimizer. Default: 2e-3.
            patience (int): Number of epochs without improvement before
                early stopping. Default: 100.
            fix_len (int): Maximum length for character/subword sequences.
                Default: 20.
            min_freq (int): Minimum word frequency to include in vocabulary.
                Default: 2.
            buckets (int): Number of buckets for length-based batching.
                Default: 1000.
            mu (float): Adam beta1 parameter. Default: 0.9.
            nu (float): Adam beta2 parameter. Default: 0.9.
            epsilon (float): Adam epsilon parameter. Default: 1e-12.
            clip (float): Gradient clipping value. Default: 5.0.
            decay (float): Learning rate decay factor. Default: 0.75.
            decay_steps (int): Number of steps between learning rate decays.
                Default: 5000.
            wandb: Optional Weights & Biases object for experiment tracking.

        Returns:
            None. The trained model is saved to `output_dir`.

        Example:
            >>> trainer.train(
            ...     output_dir='./models/my_parser',
            ...     max_epochs=100,
            ...     batch_size=5000,
            ...     lr=2e-3
            ... )
        """
        # Convert to Path and ensure it's a string for the trainer
        output_path = str(Path(output_dir))

        self._trainer.train(
            base_path=output_path,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            fix_len=fix_len,
            min_freq=min_freq,
            buckets=buckets,
            mu=mu,
            nu=nu,
            epsilon=epsilon,
            clip=clip,
            decay=decay,
            decay_steps=decay_steps,
            wandb=wandb
        )
