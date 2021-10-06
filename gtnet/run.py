from .model import load_model
from .sequence import _get_DNA_map, get_sequences, get_bidir_seq
from .utils import get_species_pred, get_label_file
import numpy as np
import pandas as pd
import logging


def predict(fasta_path, domain, vocab, **kwargs):
    if fasta_path is None:
        logging.error('Please provide a fasta path!')
        exit()

    model = load_model(domain)
    input_name = model.get_inputs()[0].name
    chars, basemap, rcmap = _get_DNA_map()
    
    species_names = get_label_file()
    
    preds = []
    for seq in get_sequences(fasta_path, basemap):
        # 1. fwd sequence is turned into bidirectional seq (chunked)
        bidir_seq = get_bidir_seq(seq, rcmap, chunk_size=4096,
                                  pad_value=8)
        # 2. pass chunks into model
        output = model.run(None, {input_name: bidir_seq.astype(np.int64)})[0]
        pred = get_species_pred(output)
        
        # 3. extract species name based off model prediction
        species = species_names['species'][pred]
        preds.append(species)
        
    pd.DataFrame(preds, columns=['species']).to_csv('predictions.csv', index=False)
