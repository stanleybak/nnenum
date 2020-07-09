'''
remove scaling from ETH networks

Stanley Bak, 2020
'''

import sys
from itertools import chain

import numpy as np
import onnxruntime as ort

import onnx
from onnx import AttributeProto, TensorProto, GraphProto
from skl2onnx.algebra.onnx_ops import OnnxTranspose, OnnxGemm, OnnxFlatten, OnnxMatMul, OnnxAdd

import matplotlib.pyplot as plt

def load_zero_images(filename, is_zero=True):
    'test reading images from csv file'

    image_list = []
    labels = []

    if 'cifar' in filename:
        mnist = False

        mean_list = [0.4914, 0.4822, 0.4465]
        sigma_list = [0.2023, 0.1994, 0.2010]
    else:
        assert 'mnist' in filename
        mnist = True

        mean = 0.1307
        sigma = 0.3081

    with open(filename, 'r') as f:
        line = f.readline()
                    
        while line is not None and len(line) > 0:
            parts = line.split(',')
            labels.append(int(parts[0]))

            if mnist:
                line_list = [0.0 if is_zero else 1.0 for _ in parts[1:]]
                image = np.array(line_list, dtype=np.float32)

                # normalization
                for i in range(image.size):
                    image[i] -= mean
                    image[i] /= sigma

                image.shape = (1, 1, 28, 28)
            else:
                #cfar load

                rgb_lists = [[], [], []]
                rgb_index = 0

                for x in parts[1:]:
                    val = 0.0 if is_zero else 1.0

                    val -= mean_list[rgb_index]
                    val /= sigma_list[rgb_index]

                    rgb_lists[rgb_index].append(val)
                    rgb_index = (rgb_index + 1) % 3

                image = np.array(rgb_lists, dtype=np.float32)

                image.shape = (1, 3, 32, 32)

            image_list.append(image)

            break
            #line = f.readline()

    return image_list, labels

def predict_with_onnxruntime(model_def, *inputs):
    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    input = {name: input for name, input in zip(names, inputs)}
    res = sess.run(None, input)
    names = [o.name for o in sess.get_outputs()]
    
    return {name: output for name, output in zip(names, res)}

def remove_unused_initializers(model):
    'return a modified model'

    new_init = []

    for init in model.graph.initializer:
        found = False
        
        for node in model.graph.node:
            for i in node.input:
                if init.name == i:
                    found = True
                    break

            if found:
                break

        if found:
            new_init.append(init)

    graph = onnx.helper.make_graph(model.graph.node, model.graph.name, model.graph.input,
                        model.graph.output, new_init)

    onnx_model = make_model_with_graph(model, graph)

    return onnx_model

def model_convert(onnx_filename):
    'make the model'

    if "cifar" in onnx_filename:
        mean_list = [0.4914, 0.4822, 0.4465]
        sigma_list = [0.2023, 0.1994, 0.2010]
    else:
        assert 'mnist' in onnx_filename
        mean_list = [0.1307]
        sigma_list = [0.3081]

    onnx_model = onnx.load(onnx_filename + ".onnx")
    onnx_model = remove_unused_initializers(onnx_model)
    
    onnx_input = onnx_model.graph.input[0].name
    print(f"onnx input: {onnx_input}")
    
    inp = onnx_model.graph.input[0]
    image_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim) # single input
    
    print(f"using input shape: {image_shape}")
    print(f"orig input shape: {tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)}")
    
    image_cols = image_shape[-1]
    image_rows = image_shape[-2]

    b_mats = []
    c_mats = []

    for mean, sigma in zip(mean_list, sigma_list):
        b_mats.append(np.identity(image_cols, dtype=np.float32) / sigma)
        c_mats.append(np.ones((image_rows, image_cols), dtype=np.float32) * (-mean))

    c_mats = np.array(c_mats)

    while len(c_mats.shape) != len(image_shape):
        c_mats = np.expand_dims(c_mats, axis=0)
    
    print(f"c_mats shape: {c_mats.shape}")
    
    b_mats = np.array(b_mats)

    while len(b_mats.shape) != len(image_shape):
        b_mats = np.expand_dims(b_mats, axis=0)
    
    print(f"b_mats shape: {b_mats.shape}")

    input_name = 'unscaled_input'
    add_node = OnnxAdd(input_name, c_mats)
    matmul_node = OnnxMatMul(add_node, b_mats, output_names=[onnx_input])

    i = onnx.helper.make_tensor_value_info('i', TensorProto.FLOAT, image_shape)

    # test matmul model
    matmul_model = matmul_node.to_onnx({input_name: i})
    onnx.checker.check_model(matmul_model)
    
    zero_in = np.zeros(image_shape, dtype=np.float32)
    o = predict_with_onnxruntime(matmul_model, zero_in)
    olist = [o[0, 0, 5, 7], o[0, 1, 5, 7], o[0, 2, 5, 7]]
    print(f"output mat_mul zeros: {olist}")

    zero_val_list = [(0 - mean) / sigma for mean, sigma in zip(mean_list, sigma_list)]
    print(f"expected zero vals = {zero_val_list}")
    assert np.allclose(olist, zero_val_list)

    one_in = np.ones(image_shape, dtype=np.float32)
    o = predict_with_onnxruntime(matmul_model, one_in)
    olist = [o[0, 0, 5, 7], o[0, 1, 5, 7], o[0, 2, 5, 7]]
    print(f"output mat_mul zeros: {olist}")
    one_val_list = [(1 - mean) / sigma for mean, sigma in zip(mean_list, sigma_list)]
    print(f"expected one vals = {one_val_list}")
    assert np.allclose(olist, one_val_list)

    print("zero and one vals matches with prefix network!")

    # only shapes are used
    model2_def = matmul_node.to_onnx({input_name: i})

    #print('The model is:\n{}'.format(model2_def))
    
    onnx.checker.check_model(model2_def)
    
    print('The models are checked!')
    combined = glue_models(model2_def, onnx_model)

    converted_filename = f"{onnx_filename}_noscale.onnx"
    onnx.save_model(combined, converted_filename)
    print(f"Saved unscaled model to: {converted_filename}\n")

def make_model_with_graph(model, graph, check_model=True):
    'copy a model with a new graph'

    onnx_model = onnx.helper.make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    # fix opset import
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    if check_model:
        onnx.checker.check_model(onnx_model, full_check=True)

    return onnx_model

def glue_models(model1, model2):
    'glue the two onnx models into one'

    graph1 = model1.graph
    graph2 = model2.graph

    assert graph1.output[0].name == graph2.input[0].name, f"graph1 output was {graph1.output[0].name}, " + \
        f"but graph2 input was {graph2.input[0].name}"

    var_in = graph1.input
    var_out = graph2.output

    combined_init = []
    names = []
    for init in chain(graph1.initializer, graph2.initializer):
        assert init.name not in names, f"repeated initializer name: {init.name}"
        names.append(init.name)

        combined_init.append(init)

    combined_nodes = []
    #names = []
    for n in chain(graph1.node, graph2.node):
        #assert n.name not in names, f"repeated node name: {n.name}"
        #names.append(n.name)

        combined_nodes.append(n)

    name = graph2.name
    graph = onnx.helper.make_graph(combined_nodes, name, var_in,
                       var_out, combined_init)

    #print(f"making model with inputs {inputs} / outputs {outputs} and nodes len: {len(keep_nodes)}")
    onnx_model = make_model_with_graph(model2, graph)

    return onnx_model

def predict_with_onnxruntime(model_def, *inputs):
    'run a model with the given inputs'
    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = {name: inp for name, inp in zip(names, inputs)}

    #print(f"inp: {inp}")
    
    return sess.run(None, inp)[0]
    #names = [o.name for o in sess.get_outputs()]
    
    #return {name: output for name, output in zip(names, res)}

def model_execute(onnx_filename):
    'execute the model and its conversion as a sanity check'

    print("Trying random input")

    converted_filename = f"{onnx_filename}_noscale.onnx"

    if "cifar" in onnx_filename:
        mean_list = [0.4914, 0.4822, 0.4465]
        sigma_list = [0.2023, 0.1994, 0.2010]
    else:
        assert 'mnist' in onnx_filename
        mean_list = [0.1307]
        sigma_list = [0.3081]

    onnx_model = onnx.load(onnx_filename + ".onnx")
    onnx_model = remove_unused_initializers(onnx_model)

    inp = onnx_model.graph.input[0]
    onnx_input = inp.name
    
    image_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim) # single input
    print(f"input shape: {image_shape}")

    np.random.seed(0)

    unscaled_i = np.random.rand(*image_shape).astype(np.float32)
    scaled_i = unscaled_i.copy()

    for channel in range(image_shape[1]):
        mean = mean_list[channel]
        sigma = sigma_list[channel]
        scaled_i[0, channel, :, :] = (unscaled_i[0, channel, :, :] - mean) / sigma

    output1 = predict_with_onnxruntime(onnx_model, scaled_i)
    print(f"output orig: {output1}")

    noscaling_model = onnx.load(converted_filename)
    output2 = predict_with_onnxruntime(noscaling_model, unscaled_i)
    print(f"output noscaling: {output2}")

    assert np.linalg.norm(output1 - output2) < 1e-4, "execution differs"

    print("random execution matches!")

if __name__ == '__main__':
    #main()
    assert len(sys.argv) == 2, f"expected single argument: <onnx-filename>, got {len(sys.argv)}"
    onnx_filename = sys.argv[1]

    assert onnx_filename.endswith(".onnx")
    onnx_filename = onnx_filename[:-5]
    
    model_convert(onnx_filename)
    model_execute(onnx_filename)
