"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log

import numpy as np

from extensions.ops.normalize import NormalizeOp
from mo.front.common.layout import get_features_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from extensions.ops.elementwise import Mul
from extensions.ops.ReduceOps import ReduceSum
from ..utils import sum_infer

class L2NormToNormPattern(MiddleReplacementPattern):
    enabled = True
    force_clean_up = False

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('bias_add', dict(op='Add', name='dense_1/BiasAdd')),
                ('merge', dict(op='Merge', name='batch_normalization_11/cond/Merge')),
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        bias_add = match['bias_add']
        merge = match['merge']
        normalize_node = NormalizeOp(graph, {'name': merge.name + '/Normalize', 
            'eps': 1e-6, 'across_spatial': 0, 'channel_shared': 0}).create_node()
        # the normalize_input_node has 2 consumers so it is necessary to disconnect output port first
        bias_add.out_port(0).connect(normalize_node.in_port(0))
        merge.in_port(0).disconnect()
        normalize_node.out_port(0).get_connection().set_destination(merge.in_port(0))
