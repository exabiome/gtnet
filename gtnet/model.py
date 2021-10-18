import os
from pkg_resources import resource_filename
import onnxruntime as rt


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
        raise ValueError("Model not available for bacteria yet")
    elif domain == 'archaea':
        path = 'ar122.onnx'
    else:
        raise ValueError("Unrecognized domain: '%s'" % domain)
    return os.path.join(resource_filename(__name__, 'models'), path)


def load_model(model_path=None, domain='archaea'):
    if model_path is None:
        model_path = _get_model_path(domain)
    model = rt.InferenceSession(model_path)
    return model
