# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" pyramidions configuration"""
from collections import OrderedDict
from typing import List, Mapping

from psutil import net_connections

from ...onnx import OnnxConfig
from ...utils import logging
from ..bert.configuration_bert import BertConfig


logger = logging.get_logger(__name__)

PYRAMIDIONS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "pyramidions": "https://huggingface.co/pyramidions/resolve/main/config.json",
}


class LayerPoolingConfigException(Exception):
    ...

class PyramidionsConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`PyramidionsModel`] or a [`TFPyramidionsModel`]. It is
    used to instantiate a pyramidions model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the pyramidions
    [pyramidions](https://huggingface.co/pyramidions) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The [`PyramidionsConfig`] class directly inherits [`BertConfig`]. It reuses the same defaults. Please check the parent
    class for more information.

    Examples:

    ```python
    >>> from transformers import PyramidionsConfig, PyramidionsModel

    >>> # Initializing a pyramidions configuration
    >>> configuration = PyramidionsConfig()

    >>> # Initializing a model from the configuration
    >>> model = PyramidionsModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "pyramidions"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, alpha: float = 1.0, encoder_layer_pooling: List[bool] = None, num_hidden_layers: int = 9, max_position_embeddings: int = 512, n_blocks: int = None, **kwargs):
        """Constructs PyramidionsConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, num_hidden_layers=num_hidden_layers, max_position_embeddings=max_position_embeddings, **kwargs)
        self.alpha = alpha
        if encoder_layer_pooling is None:
            encoder_layer_pooling = [True] * self.num_hidden_layers
        else: 
            if len(encoder_layer_pooling) != self.num_hidden_layers:
                raise LayerPoolingConfigException(
                    f"Number of hidden layers ({self.num_hidden_layers}) does not match number of entries in encoder_layer_pooling ({len(encoder_layer_pooling)})"
                    )
        num_pooling_layer = sum(encoder_layer_pooling)
        if self.max_position_embeddings * (0.5**num_pooling_layer) < 1.0:
            raise LayerPoolingConfigException(
                f"Number of layers with pooling ({num_pooling_layer}) is too high."
            )
        self.encoder_layer_pooling = encoder_layer_pooling
        self.n_blocks = n_blocks


class PyramidionsOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
