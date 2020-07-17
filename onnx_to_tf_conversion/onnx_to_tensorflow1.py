'''
convert onnx to tensorflow1 .pb models
'''

import os
import sys
import logging
import warnings

import numpy as np

import tensorflow as tf
import onnxruntime as ort

from onnx_tf.backend import prepare
import onnx

def predict_with_onnxruntime(model_def, *inputs):
    'run with onnx'
    
    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = {name: inp for name, inp in zip(names, inputs)}
    
    res = sess.run(None, inp)
    names = [o.name for o in sess.get_outputs()]
    
    return {name: output for name, output in zip(names, res)}

def main():
    'conert onnx model'

    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    assert len(sys.argv) == 2, f"expected single argument: <onnx-filename>, got {len(sys.argv)}"
    onnx_filename = sys.argv[1]
    
    print(f'loading onnx model from {onnx_filename}')
    onnx_model = onnx.load(onnx_filename)

    inp = onnx_model.graph.input[0]
    input_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim) # single input
    
    tf_rep = prepare(onnx_model)

    np.random.seed(0)
    test = np.random.rand(*input_shape).astype(np.float32)

    out = predict_with_onnxruntime(onnx_model, test)
    onnx_out = next(iter(out.values()))
    print(f"onnx result before export: {onnx_out}")

    out = tf_rep.run(test)
    print(f"tf_rep result before export: {out}")

    with tf.compat.v1.Session() as persisted_sess:
        persisted_sess.graph.as_default()
        tf.import_graph_def(tf_rep.graph.as_graph_def(), name='')

        inp = persisted_sess.graph.get_tensor_by_name(tf_rep.tensor_dict[tf_rep.inputs[0]].name)
        out = persisted_sess.graph.get_tensor_by_name(tf_rep.tensor_dict[tf_rep.outputs[0]].name)

        res = persisted_sess.run(out, {inp: test})
        print(f"tensorflow result: {res}")

        n = np.linalg.norm(res - onnx_out)
        print(f"norm of difference: {n}")
        assert n < 1e-4, "execution differs"
        print("tensorflow output matches onnx output with random input!")

    tf_rep.export_graph(f'{onnx_filename}.pb')
    print(f"tensorflow file saved to {onnx_filename}.pb")

if __name__ == '__main__':
    main()
