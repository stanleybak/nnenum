'''
nnenum vnncomp convert maxpool script

usage: "python3 convert_maxpool.py <onnx_file>"

if the onnx filename, say model.onnx, contains maxpooling layers, then DNNV will be run to create model.onnx.converted

Stanley Bak
July 2022
'''

import sys
import onnx
import time

from dnnv.nn import parse
from dnnv.nn.transformers.simplifiers import simplify, ReluifyMaxPool
#from dnnv.nn.transformers.simplifiers.reluify_maxpool import 
from pathlib import Path

def main():
    'main entry point'

    if len(sys.argv) != 2:
        print('usage: "python3 convert_maxpool.py <onnx_file>')
        sys.exit(1)

    filename = sys.argv[1]
    graph = onnx.load(filename).graph

    has_maxpool = False

    for node in graph.node:
        o = node.op_type

        if o == 'MaxPool':
            has_maxpool = True
            break

    if not has_maxpool:
        print(f"{filename} has no max pool layers to convert")
    else:
        out_filename = filename + ".converted"

        if Path(out_filename).is_file():
            print(f"{out_filename} already exists")
        else:
            print(f"creating {out_filename}...")
            t = time.perf_counter()
            op_graph = parse(Path(filename))
            simplified_op_graph1 = simplify(op_graph, reluify_maxpool(op_graph))
            simplified_op_graph1.export_onnx(out_filename)
            diff = time.perf_counter() - t
            print(f"convert runtime: {diff}")

if __name__ == '__main__':
    main()
