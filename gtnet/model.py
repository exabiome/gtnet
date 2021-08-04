import onnxruntime

# assume this is filled in and is a path to an ONNX file
PRETRAINED_MODEL_PATH = ...

def load_model(model_path=None):
    if model_path is None:
        model_path = PRETRAINED_MODEL_PATH
    # read model_path and return model
    pass
