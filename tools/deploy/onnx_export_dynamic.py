# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import sys
sys.path.append('../../')
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.file_io import PathManager
from fastreid.modeling.meta_arch import build_model
import numpy as np
import os
import argparse
import io
import cv2
import tqdm
import onnx
import onnxruntime
import onnxoptimizer
import torch
from onnxsim import simplify
from torch.onnx import OperatorExportTypes
import time

# import some modules added in project like this below
# sys.path.append('../../projects/FastDistill')
# from fastdistill import *

logger = setup_logger(name='onnx_export')


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert Pytorch to ONNX model")

    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='onnx_model',
        help='path to save converted onnx model'
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int,
        help="the maximum batch size of onnx runtime"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    logger.info("Beginning ONNX file converting")
    input_names = ['data']
    output_names = ['output']
    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                opset_version=11,
                dynamic_axes={**{input_name: {0: 'batchsize'} for input_name in input_names}, **{output_name: {0: 'batchsize'} for output_name in output_names}},
                input_names=input_names,
                output_names=output_names,
                #operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                verbose=True,  # NOTE: uncomment this for debugging
                export_params=True
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    logger.info("Completed convert of ONNX model")

    # # Apply ONNX's Optimization
    # logger.info("Beginning ONNX model path optimization")
    # all_passes = onnxoptimizer.get_available_passes()
    # passes = ["extract_constant_to_initializer",
    #           "eliminate_unused_initializer", "fuse_bn_into_conv"]
    # assert all(p in all_passes for p in passes)
    # onnx_model = onnxoptimizer.optimize(onnx_model, passes)
    # logger.info("Completed ONNX model path optimization")
    return onnx_model


def preprocess(image_path, image_height, image_width):
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height),
                     interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    if cfg.MODEL.HEADS.POOL_LAYER == 'FastGlobalAvgPool':
        cfg.MODEL.HEADS.POOL_LAYER = 'GlobalAvgPool'
    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    # if hasattr(model.backbone, 'deploy'):
    #     model.backbone.deploy(True)
    model.eval()
    model = model.cpu()
    logger.info(model)

    inputs = torch.randn(args.batch_size, 3,
                         cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1], device='cpu')
    onnx_model = export_onnx_model(model, inputs)

    #model_simp, check = simplify(onnx_model)

    #model_simp = remove_initializer_from_input(model_simp)

    #assert check, "Simplified ONNX model could not be validated"

    PathManager.mkdirs(args.output)

    save_path = os.path.join(args.output, args.name+'.onnx')
    onnx.save_model(onnx_model, save_path)
    logger.info("ONNX model file has already saved to {}!".format(save_path))
    # verify pytorch result and onnx result
    ort_sess = onnxruntime.InferenceSession(save_path)

    input_name = ort_sess.get_inputs()[0].name
    import glob
    for i in range(10):
        for path in glob.glob("test_data/*.jpg"):
            #print(path)
            image = preprocess(path, 256, 256)
            #image = np.random.randn(64,3,256,256).astype("float32")
            start = time.time()
            onnx_result = ort_sess.run(None, {input_name: image})[0]
            end = time.time()
            print(end - start)
            # feat = normalize(feat, axis=1)
            image_tensor = torch.from_numpy(image)
            with torch.no_grad():
                pytorch_result = model(image_tensor).numpy()
            #print(onnx_result.shape, pytorch_result.shape)
            #print(onnx_result, pytorch_result)
            np.testing.assert_allclose(
                onnx_result,
                pytorch_result,
                rtol=1e-03,
                atol=1e-06,
            )
