from .sequence import _get_DNA_map, batch_sequence
from .utils import get_taxon_pred, get_label_file
from .utils import get_logger, get_data_path, get_config
import onnxruntime as rt
import ruamel.yaml as yaml
import numpy as np
import pandas as pd
import argparse
import skbio
from skbio.sequence import DNA
import json
import sys
import os


def get_predictions(fasta_path, output_dest=None, **kwargs):
    if fasta_path is None:
        logging.error('The path provided is not a fasta file')
        exit()

    config = get_config()
    taxon_table = pd.read_csv(config.taxa_df_path)
    preds = []

    model =  rt.InferenceSession(config.inf_model_path)
    input_name = model.get_inputs()[0].name

    for sequence in skbio.read(fasta_path, format='fasta', 
                               constructor=DNA, validate=False):
        # 1. Turn full sequence into windowed batches
        batches = batch_sequence(sequence=sequence,
                                chars=config.chars, 
                                window=config.window, 
                                padval=config.pad_value, 
                                step=config.step)
        # 2. pass chunks into model
        output = model.run(None, {input_name: batches.astype(np.int64)})[0]
        pred_idx = get_taxon_pred(output)

        # 3. extract predicted row from taxon_table
        taxon_pred = taxon_table.iloc[pred_idx]
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
        with open(args.txt_file, 'r') as f:
            fasta_paths = [path.strip() for path in f]
    else: fasta_paths = [args.fasta_path,]
    for fasta_path in fasta_paths:
        get_predictions(fasta_path=fasta_path, 
                        output_dest=args.output)
    
    logger.info('finished')
    

def run_test(argv=None):
    data_path = get_data_path()
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str,
                        default=None, help='output destination')

    args = parser.parse_args(argv)
    get_predictions(fasta_path=data_path,   
                    output_dest=args.output)
    