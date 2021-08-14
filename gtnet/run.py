from model import load_model
from sequence import *
import onnxruntime as rt
import numpy as np
import scipy.special
import argparse

#should this be in another file?
def get_prob(x):
    soft_out = scipy.special.softmax(x)
    probabilities = soft_out.mean(axis=0)
    return probabilities

default_fasta_path = '/global/cscratch1/sd/azaidi/GCA_000006155.2_ASM615v2_genomic.fna'

def predict(fasta_path, model_path, **kwargs):
    if fasta_path is None:
        print(f'no fasta file was provided, using {default_fasta_path} to proceed')
        fasta_path = default_fasta_path

    model = load_model(model_path)
    input_name = model.get_inputs()[0].name
    
    for seq in get_sequences(fasta_path):
        # 1. chunk sequences
        bidir_seq = get_bidir_seq(seq)
        
        # 2. pass chunks into model
        output = model.run(None, {input_name: bidir_seq.astype(np.int64)})[0]
        
        # 3. get probalities
        probabilities = get_prob(output)
        

def main():
    print('starting')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--fasta_path', type = str, default= None, help = 'sequence path')
    parser.add_argument('-m', '--model_path', type = str, default = None, help = 'path to onnx model')
    
    args = parser.parse_args()
    
    predict(fasta_path=args.fasta_path, model_path=args.model_path)
    print('finished!')
    
if __name__ == '__main__':
    main()