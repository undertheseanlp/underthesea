import os
from datetime import timedelta, datetime
from pathlib import Path
from typing import Union
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from underthesea import logger, device
from underthesea.data import CoNLL
from underthesea.models.dependency_parser import BiaffineDependencyParserSupar
from underthesea.modules.model import BiaffineDependencyModel
from underthesea.utils.sp_common import pad, unk, bos
from underthesea.utils.sp_config import Config
from underthesea.utils.sp_data import Dataset
from underthesea.utils.sp_embedding import Embedding
from underthesea.utils.sp_field import Field, SubwordField
from underthesea.utils.sp_metric import Metric
from underthesea.utils.sp_parallel import DistributedDataParallel as DDP, is_master


class ParserTrainer:
    def __init__(self, parser, corpus):
        self.parser = parser
        self.corpus = corpus

    def train(
        self, base_path: Union[Path, str],
        fix_len=20,
        min_freq=2,
        buckets=32,
        batch_size=5000,
        punct=False,
        tree=False,
        proj=False,
        lr=2e-3,
        mu=.9,
        nu=.9,
        epsilon=1e-12,
        clip=5.0,
        decay=.75,
        decay_steps=5000,
        patience=100,
        verbose=True,
        max_epochs=10,
        **kwargs
    ):
        r"""
        Train any class that implement model interface

        Args:
            base_path (object): Main path to which all output during training is logged and models are saved
            max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
            verbose:
            patience:
            decay_steps:
            decay:
            clip:
            epsilon:
            nu:
            mu:
            lr:
            proj:
            tree:
            punct:
            batch_size:
            buckets:
            min_freq:
            fix_len:


        """
        ################################################################################################################
        # BUILD
        ################################################################################################################
        locals_args = {
            'base_path': base_path,
            'fix_len': fix_len,
            'min_freq': min_freq,
            'max_epochs': max_epochs
        }
        args = Config(**locals_args)
        args.feat = self.parser.embeddings
        args.embed = self.parser.embed
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            args.max_len = min(args.max_len or tokenizer.max_len, tokenizer.max_len)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
            FEAT = Field('tags', bos=bos)

        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        if args.feat in ('char', 'bert'):
            transform = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
        else:
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)

        train = Dataset(transform, self.corpus.train)
        WORD.build(train, min_freq, (Embedding.load(args.embed, unk) if self.parser.embed else None))
        FEAT.build(train)
        REL.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_rels': len(REL.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'feat_pad_index': FEAT.pad_index,
            'device': device,
            'path': base_path
        })
        model = BiaffineDependencyModel(**args)
        model.load_pretrained(WORD.embed).to(device)
        parser_supar = BiaffineDependencyParserSupar(args, model, transform)

        ################################################################################################################
        # TRAIN
        ################################################################################################################
        args = Config()
        args.update({
            'train': self.corpus.train,
            'dev': self.corpus.dev,
            'test': self.corpus.test
        })
        parser_supar.transform.train()
        parser_supar.args.clip = clip
        parser_supar.args.punct = punct
        parser_supar.args.tree = tree
        parser_supar.args.proj = proj
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info("Loading the data")
        train = Dataset(parser_supar.transform, self.corpus.train, **args)
        dev = Dataset(parser_supar.transform, self.corpus.dev)
        test = Dataset(parser_supar.transform, self.corpus.test)
        train.build(batch_size, buckets, True, dist.is_initialized())
        dev.build(batch_size, buckets)
        test.build(batch_size, buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        logger.info(f"{parser_supar.model}\n")
        if dist.is_initialized():
            parser_supar.model = DDP(parser_supar.model,
                                     device_ids=[dist.get_rank()],
                                     find_unused_parameters=True)
        parser_supar.optimizer = Adam(parser_supar.model.parameters(),
                                      lr,
                                      (mu, nu),
                                      epsilon)
        parser_supar.scheduler = ExponentialLR(parser_supar.optimizer, decay ** (1 / decay_steps))

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, max_epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {max_epochs}:")
            parser_supar._train(train.loader)
            loss, dev_metric = parser_supar._evaluate(dev.loader)
            logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = parser_supar._evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    parser_supar.save(base_path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if epoch - best_e >= patience:
                break
        loss, metric = parser_supar.load(base_path)._evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} - {best_metric}")
        logger.info(f"{'test:':6} - {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")
