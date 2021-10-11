import os
from pkg_resources import resource_filename
import onnxruntime as rt
import requests


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
        path = "ar122.onnx"
    else:
        raise ValueError("Unrecognized domain: '%s'" % domain)
    return os.path.join(resource_filename(__name__, 'models'), path)


def load_model(model_path=None, domain='archea'):
    if model_path is None:
        model_path = _get_model_path(domain=domain)
    model = rt.InferenceSession(model_path)
    return model


def download_models(argv=None):
    required = [
        {'path': 'gtnet/models/ar122.onnx',
         'url': 'https://osf.io/yu738/download'}
    ]
    for d in required:
        if not os.path.exists(d['path']):
            print(f'Downloading {d["path"]} from {d["url"]}')
            r = requests.get(d['url'], allow_redirects=True)
            with open(d['path'], 'wb') as f:
                f.write(r.content)
        else:
            print(f'{d["path"]} already exists, skipping download')
