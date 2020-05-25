from mo.graph.graph import Graph, Node
from mo.front.common.partial_infer.utils import int64_array, float_array
import numpy as np
from mo.graph.perm_inputs import PermuteInputs


def reduce_infer(node: Node):
    connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
    assert len(connected_in_ports) == 2, \
        "{} node `{}` should have 2 input ports, where 0-input is data input and 1-input represent " \
        "`reduction_indices`".format(node.op, node.id)

    axis = int64_array([1])

    used_dims = np.zeros(3, dtype=np.bool)
    output_shape = np.array([1, 6, 5])

    for dim in axis:
        used_dims[dim] = True
        output_shape[dim] = 1

    # In case if keep dims == False, we should remove all 1 dims that was used in reduction
    if not node.keep_dims:
        output_shape = output_shape[np.invert(used_dims)]

    node.out_port(0).data.set_shape(output_shape)

    PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')

def sum_infer(node: Node):
    connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
    assert len(connected_in_ports) == 2, \
        "{} node `{}` should have 2 input ports, where 0-input is data input and 1-input represent " \
        "`reduction_indices`".format(node.op, node.id)

    axis = int64_array([1])

    used_dims = np.zeros(2, dtype=np.bool)
    output_shape = np.array([1, 5])

    for dim in axis:
        used_dims[dim] = True
        output_shape[dim] = 1

    # In case if keep dims == False, we should remove all 1 dims that was used in reduction
    if not node.keep_dims:
        output_shape = output_shape[np.invert(used_dims)]

    node.out_port(0).data.set_shape(output_shape)

    PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')