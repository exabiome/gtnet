from model import load_model
from sequence import *
import onnxruntime as rt
import numpy as np
import scipy.special

#should this be in another file?
def get_prob(x):
    soft_out = scipy.special.softmax(x)
    probabilities = soft_out.mean(axis=0)
    return probabilities

def predict(fasta_path, model_path, **kwargs):
    if fasta_path is None:
        raise ValueError('sequences must not be None')

    model = load_model(model_path)
    input_name = model.get_inputs()[0].name
    
    for seq in get_sequences(fasta_path):
        # 1. chunk sequences
        bidir_seq = get_bidir_seq(seq)
        print(bidir_seq.shape)
        # 2. pass chunks into model
        output = model.run(None, {input_name: bidir_seq.astype(np.int64)})[0]
        probabilities = get_prob(output)
        print(probabilities.shape)
        

def main():
    print('starting')
    fasta_path = '/global/cscratch1/sd/azaidi/GCA_000006155.2_ASM615v2_genomic.fna'
    model_path = 'data/bac120_r202.resnet50.genus.onnx'
    predict(fasta_path=fasta_path, model_path=model_path)
    
if __name__ == '__main__':
    main()