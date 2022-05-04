# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert pyramidions checkpoint."""


import argparse
import pathlib

import fairseq
import torch
from fairseq.models.pyramidions import PyramidionsModel as FairseqPyramidionsModel
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version

from transformers import PyramidionsConfig, PyramidionsForMaskedLM, PyramidionsForSequenceClassification
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.utils import logging


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"


def convert_pyramidions_checkpoint_to_pytorch(
    pyramidions_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    """
    Copy/paste/tweak pyramidions's weights to our BERT structure.
    """
    pyramidions = FairseqPyramidionsModel.from_pretrained(pyramidions_checkpoint_path)
    pyramidions.eval()  # disable dropout
    pyramidions_sent_encoder = pyramidions.model.encoder.sentence_encoder
    config = PyramidionsConfig(
        vocab_size=pyramidions_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=pyramidions.args.encoder_embed_dim,
        num_hidden_layers=pyramidions.args.encoder_layers,
        num_attention_heads=pyramidions.args.encoder_attention_heads,
        intermediate_size=pyramidions.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )
    if classification_head:
        config.num_labels = pyramidions.model.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our BERT config:", config)

    model = PyramidionsForSequenceClassification(config) if classification_head else PyramidionsForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    model.pyramidions.embeddings.word_embeddings.weight = pyramidions_sent_encoder.embed_tokens.weight
    model.pyramidions.embeddings.position_embeddings.weight = pyramidions_sent_encoder.embed_positions.weight
    model.pyramidions.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.pyramidions.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c pyramidions doesn't use them.
    model.pyramidions.embeddings.LayerNorm.weight = pyramidions_sent_encoder.emb_layer_norm.weight
    model.pyramidions.embeddings.LayerNorm.bias = pyramidions_sent_encoder.emb_layer_norm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.pyramidions.encoder.layer[i]
        pyramidions_layer: TransformerSentenceEncoderLayer = pyramidions_sent_encoder.layers[i]

        # self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert (
            pyramidions_layer.self_attn.k_proj.weight.data.shape
            == pyramidions_layer.self_attn.q_proj.weight.data.shape
            == pyramidions_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = pyramidions_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = pyramidions_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = pyramidions_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = pyramidions_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = pyramidions_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = pyramidions_layer.self_attn.v_proj.bias

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == pyramidions_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = pyramidions_layer.self_attn.out_proj.weight
        self_output.dense.bias = pyramidions_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = pyramidions_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = pyramidions_layer.self_attn_layer_norm.bias

        # intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == pyramidions_layer.fc1.weight.shape
        intermediate.dense.weight = pyramidions_layer.fc1.weight
        intermediate.dense.bias = pyramidions_layer.fc1.bias

        # output
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == pyramidions_layer.fc2.weight.shape
        bert_output.dense.weight = pyramidions_layer.fc2.weight
        bert_output.dense.bias = pyramidions_layer.fc2.bias
        bert_output.LayerNorm.weight = pyramidions_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = pyramidions_layer.final_layer_norm.bias
        # end of layer

    if classification_head:
        model.classifier.dense.weight = pyramidions.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = pyramidions.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = pyramidions.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = pyramidions.model.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = pyramidions.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = pyramidions.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = pyramidions.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = pyramidions.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = pyramidions.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = pyramidions.model.encoder.lm_head.bias

    # Let's check that we get the same results.
    input_ids: torch.Tensor = pyramidions.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = pyramidions.model.classification_heads["mnli"](pyramidions.extract_features(input_ids))
    else:
        their_output = pyramidions.model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pyramidions_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    args = parser.parse_args()
    convert_pyramidions_checkpoint_to_pytorch(
        args.pyramidions_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
