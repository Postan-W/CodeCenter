"""
Date: 2022/05/13
"""
import onnx
from onnx_tf.backend import prepare
print("onnx====>>pb")
onnx_model = onnx.load("./flowers_cli_dense2onnx.onnx")
tf_exp = prepare(onnx_model)  # prepare tf representation
tf_exp.export_graph("./tensorflowmodel.pb")