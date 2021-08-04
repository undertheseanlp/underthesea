from os.path import join
from underthesea.datasets.vlsp2020_dp import VLSP2020_DP
from underthesea.file_utils import MODELS_FOLDER
from underthesea.models.dependency_parser import DependencyParser
from underthesea.modules.embeddings import FieldEmbeddings
from underthesea.trainers.dependency_parser_trainer import DependencyParserTrainer
import wandb

wandb.init(project='vlsp2020-dp', entity='rain1024')
corpus = VLSP2020_DP()

embeddings = [
    FieldEmbeddings()
]
parser = DependencyParser(
    embeddings=embeddings, init_pre_train=True,
    feat='bert',
    bert='vinai/phobert-base',
    n_feat_embed=768)
trainer = DependencyParserTrainer(parser, corpus)
trainer.train(
    base_path=join(MODELS_FOLDER, 'parsers', 'vi-dp-v1.3.2a2'),
    max_epochs=1000,
    batch_size=10000,
    mu=.9,  # optimizer parameters
    wandb=wandb
)
