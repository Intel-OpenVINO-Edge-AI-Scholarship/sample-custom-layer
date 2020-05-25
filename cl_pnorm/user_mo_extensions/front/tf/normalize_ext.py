"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op
from extensions.back.ReduceToPooling import ReduceReplacer

# inverse power for ratios
def get_power_attr():
    return -1

# group multiplicity for reshape
def get_group_reshape():
    return 1

def const_average_ratio():
    return 0.167

def significant_digits():
    return [1, 4]

class ResultFrontExtractor(FrontExtractorOp):
    kind = 'op'
    op = 'Merge'
    name = 'batch_normalization_11/cond/Merge'
    enabled = True
    force_clean_up = False

    @staticmethod
    def extract(node):
      digits = significant_digits()
      attrs = {
        'op': "Merge",
        'p': get_power_attr(),
        'group': get_group_reshape(),
        'significant': digits[0],
        'to_significant': digits[1],
        'const_avg_ratio': const_average_ratio()
      }

      # update the attributes of the node
      Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)
      ReduceReplacer.enabled = False
      return __class__.enabled
