
import torch
import torch.nn as nn
from torch import optim

# 兼容性修复：在Python 3.10+中，Mapping和Sequence已从collections移到collections.abc
import collections
from collections.abc import Mapping, Sequence
collections.Mapping = Mapping
collections.Sequence = Sequence

# 兼容性修复：numpy 2.0.0中移除了np.Inf别名，只支持np.inf
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import pytorch_lightning as pl
from torchmetrics.functional import f1_score

from .model import HallucinationBaseline, TransformerBaseline, GroundedBaseline


class Experiment(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['model']['name'] == 'TransformerBaseline':
            self.model = TransformerBaseline(config.get('model', {}))
        elif config['model']['name'] == 'GroundedBaseline':
            self.model = GroundedBaseline(config.get('model', {}))
        elif config['model']['name'] == 'HallucinationBaseline':
            print('surprise!!!')
            self.model = HallucinationBaseline(config.get('model', {}))
        self.save_hyperparameters(config)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_index):
        output = self(batch)
        return {
            "loss": output.loss,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_index):
        labels = batch.pop('labels')
        output = self(batch)
        return {
            'gold': labels,
            'pred': output.logits.argmax(dim=1),
        }

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)
        prob = output.logits.softmax(dim=1)
        prob = prob[:, 1].tolist()
        indexes = batch['indexes'].tolist()
        return {
            'predictions': prob,
            'indexes': indexes,
        }
    
    def validation_epoch_end(self, outputs):
        pred = torch.cat([x['pred'] for x in outputs])
        gold = torch.cat([x['gold'] for x in outputs])
        f1 = f1_score(pred, gold, task='binary')  # 注意加 task 参数
        #pred_list = [x['pred'].cpu() for x in outputs]
        #gold_list = [x['gold'].cpu() for x in outputs]
        #pred = torch.cat(pred_list)
        #gold = torch.cat(gold_list)
        self.log('f1',f1, prog_bar=True)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        return optim.AdamW(params, lr=self.config['lr'])