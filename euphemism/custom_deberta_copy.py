from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Layer,
    ConvLayer,
    ContextPooler,
    StableDropout,
    build_relative_position,
    DebertaV2PreTrainedModel,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    SequenceClassifierOutput,
)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------
class CustomDebertaV2Embeddings(nn.Module):
    """Construct embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            self.embedding_size,
            padding_idx=pad_token_id,
        )

        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, self.embedding_size)
            if self.position_biased_input
            else None
        )

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size,
                self.embedding_size,
            )

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(
                self.embedding_size,
                config.hidden_size,
                bias=False,
            )

        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )
        self.config = config

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape,
                dtype=torch.long,
                device=self.position_ids.device,
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings(position_ids)

        if self.config.type_vocab_size > 0:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            embeddings = embeddings * mask.to(embeddings.dtype)

        embeddings = self.dropout(embeddings)
        return embeddings


# ---------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------
class CustomDebertaV2Encoder(nn.Module):
    """DeBERTa-V2 encoder with relative position bias."""

    def __init__(self, config):
        super().__init__()

        self.layer = nn.ModuleList(
            [DebertaV2Layer(config) for _ in range(config.num_hidden_layers)]
        )

        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(
                config, "max_relative_positions", config.max_position_embeddings
            )
            self.position_buckets = getattr(config, "position_buckets", -1)

            size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(size, config.hidden_size)

        self.norm_rel_ebd = [
            x.strip()
            for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")
        ]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(
                config.hidden_size,
                config.layer_norm_eps,
                elementwise_affine=True,
            )

        self.conv = (
            ConvLayer(config)
            if getattr(config, "conv_kernel_size", 0) > 0
            else None
        )

        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        if not self.relative_attention:
            return None
        rel = self.rel_embeddings.weight
        if "layer_norm" in self.norm_rel_ebd:
            rel = self.LayerNorm(rel)
        return rel

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = mask * mask.squeeze(-2).unsqueeze(-1)
            return mask.byte()
        return attention_mask.unsqueeze(1)

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        input_mask = (
            attention_mask
            if attention_mask.dim() <= 2
            else (attention_mask.sum(-2) > 0).byte()
        )

        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        next_kv = hidden_states[0] if isinstance(hidden_states, Sequence) else hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states += (output_states,)

            output_states = layer_module(
                next_kv,
                attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
            )

            if output_attentions:
                output_states, attn = output_states
                all_attentions += (attn,)

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            next_kv = output_states
            if query_states is not None:
                query_states = output_states

        if output_hidden_states:
            all_hidden_states += (output_states,)

        if not return_dict:
            return tuple(
                v for v in (output_states, all_hidden_states, all_attentions) if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# ---------------------------------------------------------------------
# Base Model
# ---------------------------------------------------------------------
class CustomDebertaV2Model(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = CustomDebertaV2Embeddings(config)
        self.encoder = CustomDebertaV2Encoder(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        visual_features=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        if visual_features is not None:
            embedding_output = torch.cat([visual_features, embedding_output], dim=1)
            ones = torch.ones(
                visual_features.size()[:2],
                device=embedding_output.device,
            )
            attention_mask = torch.cat([ones, attention_mask], dim=1)

        return self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )


# ---------------------------------------------------------------------
# Sequence Classification (Single-Label, Multi-Class)
# ---------------------------------------------------------------------
class CustomDebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.deberta = CustomDebertaV2Model(config)
        self.pooler = ContextPooler(config)

        self.classifier = nn.Linear(self.pooler.output_dim, self.num_labels)
        self.dropout = StableDropout(config.hidden_dropout_prob)

        # ⭐ 修改点 1：明确声明单标签任务
        self.config.problem_type = "single_label_classification"

        self.post_init()

    def forward(
        self,
        input_ids=None,
        visual_features=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids=input_ids,
            visual_features=visual_features,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.pooler(outputs.last_hidden_state)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [B, C]

        loss = None
        if labels is not None:
            # ⭐ 修改点 2：CrossEntropyLoss
            # labels: [B], dtype=long
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            return ((loss, logits) if loss is not None else (logits,))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

