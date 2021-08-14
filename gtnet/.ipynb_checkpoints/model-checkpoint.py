import onnxruntime as rt

# assume this is filled in and is a path to an ONNX file
PRETRAINED_MODEL_PATH = 'data/bac120_r202.resnet50.genus.onnx'

# read model_path and return model
def load_model(model_path=None):
    if model_path is None:
        print(f'a model path was not provided, using {PRETRAINED_MODEL_PATH} to proceed')
        model_path = PRETRAINED_MODEL_PATH
    model = rt.InferenceSession(model_path)
    return model


