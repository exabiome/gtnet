import os
from pkg_resources import resource_filename
import onnxruntime as rt


# assume this is filled in and is a path to an ONNX file
PRETRAINED_MODEL_PATH = 'models/bac120_r202.resnet50.genus.onnx'


def _get_model_path(domain):
    """
    Get the path to the ONNX model file for a given domain

    Parameters
    -----------
    domain: {'bacteria', 'archaea'}
        The string indication which model to get the path for

    Returns
    -------
    path: str
        The absolute path to the ONNX model file
    """
    if domain == 'bacteria':
        path = "bac120_r202.resnet50.genus.onnx"
    elif domain == 'archaea':
        raise ValueError("Model not available for archaea yet")
    else:
        raise ValueError("Unrecognized domain: '%s'" % domain)
    return os.path.join(resource_filename(__name__, 'models'), path)


def load_model(model_path=None):
    if model_path is None:
        print(f'a model path was not provided, \
                using {PRETRAINED_MODEL_PATH} to proceed')
        model_path = PRETRAINED_MODEL_PATH
    model = rt.InferenceSession(model_path)
    return model
