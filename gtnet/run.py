from model import load_model
from sequence import _get_DNA_map, get_sequences, get_bidir_seq
import numpy as np
import argparse
import logging


def predict(fasta_path, model_path, **kwargs):
    if fasta_path is None:
        logging.error('Please provide a fasta path!')
        exit()

    model = load_model(model_path)
    input_name = model.get_inputs()[0].name
    chars, basemap = _get_DNA_map()
    rcmap = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17,
                      0,  1,  2,  3,  4,  5,  6,  7,  8])

    for seq in get_sequences(fasta_path, basemap):
        # 1. chunk sequences
        bidir_seq = get_bidir_seq(seq, rcmap)
        # 2. pass chunks into model
        output = model.run(None, {input_name: bidir_seq.astype(np.int64)})[0]


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(levelname)s-%(message)s')
    logging.info('starting')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta_path', type=str,
                        default=None, help='sequence path')
    parser.add_argument('-m', '--model_path', type=str,
                        default=None, help='path to onnx model')
    args = parser.parse_args()
    predict(fasta_path=args.fasta_path, model_path=args.model_path)
    logging.info('finished!')


if __name__ == '__main__':
    main()
