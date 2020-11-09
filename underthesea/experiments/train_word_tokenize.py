from underthesea.datasets.vlsp2013_wtk_r2 import VLSP2013_WTK_R2
from underthesea.models.crf_sequence_tagger import CRFSequenceTagger
from underthesea.trainers import ModelTrainer

features = []
tagger = CRFSequenceTagger(features)
corpus = VLSP2013_WTK_R2()
trainer = ModelTrainer(tagger, corpus)

trainer.train("models/wtk_crf")
