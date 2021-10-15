from .model import load_model
from .sequence import _get_DNA_map, get_sequences, get_bidir_seq
from .utils import get_species_pred, get_label_file
import numpy as np
import pandas as pd
import logging
import argparse


def get_predictions(fasta_path, domain, vocab, output_dest, **kwargs):
    if fasta_path is None:
        logging.error('Please provide a fasta path!')
        exit()

    model = load_model(domain=domain)
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

    final_df = pd.DataFrame(preds, columns=['species'])
    if output_dest:
        final_df.to_csv(f'{output_dest}/predictions.csv', index=False)
    else:
        print(final_df)
                                                         

def predict(argv=None):
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(levelname)s-%(message)s')
    logging.info('starting')

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta_path', type=str,
                        default=None, help='sequence path')
    parser.add_argument('-d', '--domain', type=str,
                        default='archaea', help='domain',
                        choices=['bacteria', 'archaea'])
    parser.add_argument('-v', '--vocab', type=str,
                        default=None, help='vocabulary')
    parser.add_argument('-o', '--output', type=str,
                        default=None, help='output destination')
    args = parser.parse_args(argv)

    get_predictions(fasta_path=args.fasta_path, domain=args.domain,
                    vocab=args.vocab, output_dest=args.output)

    logging.info('finished!')
