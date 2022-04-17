from .model import load_model
from .sequence import _get_DNA_map, get_sequences, get_bidir_seq
from .utils import get_species_pred, get_label_file
from .utils import get_logger, get_data_path
import numpy as np
import pandas as pd
import argparse
import sys


def get_predictions(fasta_path, domain, vocab, output_dest=None, **kwargs):
    if fasta_path is None:
        logging.error('The path provided is not a fasta file')
        exit()

    model = load_model()
    input_name = model.get_inputs()[0].name
    chars, basemap, rcmap = _get_DNA_map()

    taxon_table = get_label_file()
    preds = []

    for sequence in skbio.read(fasta_path, format='fasta', constructor=DNA, validate=False):
        # 1. Turn full sequence into windowed batches
        batches = batch_sequence(sequence, window, padval, step)
        # 2. pass chunks into model
        output = model.run(None, {input_name: batches.astype(np.int64)})[0]
        species_pred = get_species_pred(output)

        # 3. extract predicted row from taxon_table
        taxon_pred = taxon_table[taxon_table.species == species_pred]
        preds.append(taxon_pred)

    final_df = pd.DataFrame(preds, columns=taxon_table.columns)
    if output_dest:
        final_df.to_csv(output_dest, index=False)
    else:
        final_df.to_csv(sys.stdout, index=False)
                                                         

def predict(argv=None):
    logger = get_logger()
    logger.info('starting')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta_path', type=str,
                        default=None, help='sequence path')
    parser.add_argument('-t', '--txt_file', type=str,
                        default=None, help='txt file with fasta paths')
    parser.add_argument('-o', '--output', type=str,
                        default=None, help='output destination')
    args = parser.parse_args(argv)

    if(args.txt_file):
        fasta_paths = #read in fasta paths
    else: fasta_paths = [args.fasta_path,]
    for fasta_path in fasta_paths:
        get_predictions(fasta_path=fasta_path, vocab=args.vocab, 
                        output_dest=args.output)
    
    logger.info('finished')
    

def run_test(argv=None):
    data_path = get_data_path()
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str,
                        default=None, help='output destination')

    args = parser.parse_args(argv)
    get_predictions(fasta_path=data_path, vocab=args.vocab, 
                    output_dest=args.output)
    