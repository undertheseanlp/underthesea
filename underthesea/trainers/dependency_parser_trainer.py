import os
from datetime import timedelta, datetime
from pathlib import Path
from typing import Union
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from underthesea.transforms.conll import CoNLL, progress_bar
from underthesea.models.dependency_parser import DependencyParser
from underthesea.utils import device, logger
from underthesea.utils.sp_common import pad, unk, bos
from underthesea.utils.sp_data import Dataset
from underthesea.utils.sp_embedding import Embedding
from underthesea.utils.sp_field import Field, SubwordField
from underthesea.utils.sp_metric import Metric, AttachmentMetric
from underthesea.utils.sp_parallel import DistributedDataParallel as DDP, is_master


class DependencyParserTrainer:
    def __init__(self, parser, corpus):
        self.parser = parser
        self.corpus = corpus

    # flake8: noqa: C901
    def train(
        self, base_path: Union[Path, str],
        fix_len=20,
        min_freq=2,
        buckets=1000,
        batch_size=5000,
        lr=2e-3,
        mu=.9,
        nu=.9,
        epsilon=1e-12,
        clip=5.0,
        decay=.75,
        decay_steps=5000,
        patience=100,
        max_epochs=10,
        wandb=None
    ):
        r"""
        Train any class that implement model interface

        Args:
            base_path (object): Main path to which all output during training is logged and models are saved
            max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
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
            batch_size:
            buckets:
            min_freq:
            fix_len:


        """
        ################################################################################################################
        # BUILD
        ################################################################################################################
        feat = self.parser.feat
        embed = self.parser.embed
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        if feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=fix_len)
        elif feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.parser.bert)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
            FEAT = Field('tags', bos=bos)

        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        if feat in ('char', 'bert'):
            transform = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
        else:
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)

        train = Dataset(transform, self.corpus.train)
        WORD.build(train, min_freq, (Embedding.load(embed, unk) if self.parser.embed else None))
        FEAT.build(train)
        REL.build(train)
        n_words = WORD.vocab.n_init
        n_feats = len(FEAT.vocab)
        n_rels = len(REL.vocab)
        pad_index = WORD.pad_index
        unk_index = WORD.unk_index
        feat_pad_index = FEAT.pad_index
        parser = DependencyParser(
            n_words=n_words,
            n_feats=n_feats,
            n_rels=n_rels,
            pad_index=pad_index,
            unk_index=unk_index,
            feat_pad_index=feat_pad_index,
            transform=transform,
            feat=self.parser.feat,
            bert=self.parser.bert
        )
        # word_field_embeddings = self.parser.embeddings[0]
        # word_field_embeddings.n_vocab = 100
        parser.embeddings = self.parser.embeddings
        # parser.embeddings[0] = word_field_embeddings
        parser.load_pretrained(WORD.embed).to(device)

        ################################################################################################################
        # TRAIN
        ################################################################################################################
        if wandb:
            wandb.watch(parser)
        parser.transform.train()
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info('Loading the data')
        train = Dataset(parser.transform, self.corpus.train)
        dev = Dataset(parser.transform, self.corpus.dev)
        test = Dataset(parser.transform, self.corpus.test)
        train.build(batch_size, buckets, True, dist.is_initialized())
        dev.build(batch_size, buckets)
        test.build(batch_size, buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")
        logger.info(f'{parser}')
        if dist.is_initialized():
            parser = DDP(parser, device_ids=[dist.get_rank()], find_unused_parameters=True)

        optimizer = Adam(parser.parameters(), lr, (mu, nu), epsilon)
        scheduler = ExponentialLR(optimizer, decay ** (1 / decay_steps))

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, max_epochs + 1):
            start = datetime.now()
            logger.info(f'Epoch {epoch} / {max_epochs}:')

            parser.train()

            bar = progress_bar(train.loader)
            metric = AttachmentMetric()
            for words, feats, arcs, rels in bar:
                optimizer.zero_grad()

                mask = words.ne(parser.WORD.pad_index)
                # ignore the first token of each sentence
                mask[:, 0] = 0
                s_arc, s_rel = parser.forward(words, feats)
                loss = parser.forward_loss(s_arc, s_rel, arcs, rels, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(parser.parameters(), clip)
                optimizer.step()
                scheduler.step()

                arc_preds, rel_preds = parser.decode(s_arc, s_rel, mask)
                # ignore all punctuation if not specified
                if not self.parser.args['punct']:
                    mask &= words.unsqueeze(-1).ne(parser.puncts).all(-1)
                metric(arc_preds, rel_preds, arcs, rels, mask)
                bar.set_postfix_str(f'lr: {scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}')

            dev_loss, dev_metric = parser.evaluate(dev.loader)
            logger.info(f"{'dev:':6} - loss: {dev_loss:.4f} - {dev_metric}")
            test_loss, test_metric = parser.evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {test_loss:.4f} - {test_metric}")
            if wandb:
                wandb.log({"test_loss": test_loss})
                wandb.log({"test_metric_uas": test_metric.uas})
                wandb.log({"test_metric_las": test_metric.las})

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    parser.save(base_path)
                logger.info(f'{t}s elapsed (saved)\n')
            else:
                logger.info(f'{t}s elapsed\n')
            elapsed += t
            if epoch - best_e >= patience:
                break
        loss, metric = parser.load(base_path).evaluate(test.loader)

        logger.info(f'Epoch {best_e} saved')
        logger.info(f"{'dev:':6} - {best_metric}")
        logger.info(f"{'test:':6} - {metric}")
        logger.info(f'{elapsed}s elapsed, {elapsed / epoch}s/epoch')
