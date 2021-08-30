from model import load_model
from sequence import _get_DNA_map, get_sequences, get_bidir_seq
import numpy as np
import logging


def predict(fasta_path, model_path, vocab, **kwargs):
    if fasta_path is None:
        logging.error('Please provide a fasta path!')
        exit()

    model = load_model(model_path)
    input_name = model.get_inputs()[0].name
    chars, basemap, rcmap = _get_DNA_map()

    for seq in get_sequences(fasta_path, basemap):
        # 1. chunk sequences
        bidir_seq = get_bidir_seq(seq, rcmap, chunk_size=4096,
                                  pad_value=8)
        # 2. pass chunks into model
        output = model.run(None, {input_name: bidir_seq.astype(np.int64)})[0]