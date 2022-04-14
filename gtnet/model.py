import os
from pkg_resources import resource_filename
import onnxruntime as rt
import requests


def _get_model_path(model_type):
    """
    Get the path to appropriate ONNX model files
    Parameters
    -----------
    model_type: {'prediciton_model', 'confidence_model'}
        The string indication which model to get the path for
    
    Returns
    -------
    path: str
        The absolute path to the ONNX model file
    """
    deploy_path = 'gtnet.deploy'
    if(os.path.exists(deploy_path)):
        return os.path.join(resource_filename(__name__, 'gtnet.deploy'), model_type)    
    else:
        raise ValueError(f"The {model_type} model is not available")


def load_model(model_type):
    '''
    This should return onnx runtime inference session of the
    onnx checkpoint specified
    '''
    model_path = _get_model_path(model_type)
    model = rt.InferenceSession(model_path)
    return model
