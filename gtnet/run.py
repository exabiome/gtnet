from .sequence import _get_DNA_map, batch_sequence, DNAEncoder
from .utils import get_taxon_pred, get_config
from .utils import get_logger, get_data_path
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
from pkg_resources import resource_filename


def get_predictions(fasta_path, output_dest=None, **kwargs):
    '''
    Extracts associated configuration file for model to
    instantiate model, label file and encoder

    Takes a Fasta file and with each sequence in the file it
    will break them into batches, run inference on that sequence
    and determine the best prediction.

    Records all predictions into a single DataFrame that is
    written to `output_dest` or standard output if `output_dest` is not provided

    Parameters
    ----------
    fasta_path : str
        Path of single fasta file

    output_dest : str, default=None
        Path where final predictions will be deposited
    '''
    if fasta_path is None:
        logging.error('The path provided is not a fasta file')
        exit()

    config = get_config()
    taxon_table = pd.read_csv(config.taxa_df_path)
    preds = []

    model =  rt.InferenceSession(config.inf_model_path)
    input_name = model.get_inputs()[0].name
    encoder = DNAEncoder(config.chars)

    for sequence in skbio.read(fasta_path, format='fasta',
                               constructor=DNA, validate=False):
        # 1. Turn full sequence into windowed batches
        batches = batch_sequence(sequence=sequence,
                                window=config.window,
                                padval=config.pad_value,
                                step=config.step,
                                encoder=encoder)
        batches = batches[:10]
        # 2. pass chunks into model
        output = model.run(None, {input_name: batches.astype(np.int64)})[0]
        pred_idx = get_taxon_pred(output)

        # 3. extract predicted row from taxon_table
        taxon_pred = taxon_table.iloc[pred_idx]
        preds.append(taxon_pred)



def _load_deploy_pkg():
    deploy_dir = resource_filename(__name__, 'deploy_pkg')

    files = ('taxa_table', 'nn_model', 'conf_model', 'training_config')
    # read manifest
    with open(os.path.join(deploy_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    # remap files in deploy_dir to be relative to where we are running
    for key in files:
        manifest[key] = os.path.join(deploy_dir, manifest[key])

    model_path = manifest['nn_model']
    conf_model_path = manifest['conf_model']
    config_path = manifest['training_config']
    tt_path = manifest['taxa_table']
    vocab = "".join(manifest['vocabulary'])

    return model_path, conf_model_path, config_path, tt_path, vocab

def run_onnx_inference(argv=None):
    """
    Convert a Torch model checkpoint to ONNX format
    """

    import argparse
    import json
    import logging
    from time import time

    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view as swv
    import onnxruntime as ort
    import pandas as pd
    import ruamel.yaml as yaml
    import skbio
    from skbio.sequence import DNA


    from .sequence import FastaReader, FastaSequenceEncoder

    desc = "Run predictions using ONNX"
    epi = ("")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('fastas', nargs='+', type=str, help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-c', '--n_chunks', type=int, default=10000, help='the number of sequence chunks to process at a time')
    parser.add_argument('-F', '--fof', action='store_true', default=False, help='a file-of-files was passed in')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output file to save classifications to')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print specific information about sequences')

    args = parser.parse_args(argv)

    if args.fof:
        with open(args.fastas[0], 'r') as f:
            args.fastas = [s.strip() for s in f.readlines()]

    logger = get_logger()
    if args.debug:
        logger.setLevel(logging.DEBUG)


    model_path, conf_model_path, config_path, tt_path, vocab = _load_deploy_pkg()

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    logger.info(f'loading model from {model_path}')
    nn_sess = ort.InferenceSession(model_path, sess_options=so, providers=['CUDAExecutionProvider'])

    logger.info(f'loading confidence model from {conf_model_path}')
    conf_sess = ort.InferenceSession(conf_model_path, sess_options=so, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    n_max_probs = conf_sess.get_inputs()[0].shape[1] - 1

    logger.info(f'loading training config from {config_path}')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f'loading taxonomy table from {tt_path}')
    tt_df = pd.read_csv(tt_path)
    outcols = ['filename', 'ID'] + tt_df.columns.tolist() + ['score']
    outcols.remove('taxon_id')

    assert nn_sess.get_outputs()[0].shape[1] == len(tt_df)

    logger.info(f'found {len(tt_df)} taxa')

    window = config['window']
    step = config['step']
    k = len(tt_df) - n_max_probs

    all_preds = list()
    all_maxprobs = list()
    all_lengths = list()
    all_seqnames = list()
    all_filepaths = list()
    aggregated = list()

    logger.info(f'beginning inference')
    before = time()

    encoder = FastaSequenceEncoder(window, step, vocab=vocab)
    reader = FastaReader(encoder)

    n_seqs = 0
    for file_path, seq_name, seq_len, seq_chunks in reader.read(args.fastas):
        all_seqnames.append(seq_name)
        all_lengths.append(seq_len)
        all_filepaths.extend(file_path)

        logger.debug(f'getting outputs for {all_seqnames[-1]}, {len(seq_chunks)} chunks, {all_lengths[-1]} bases')
        outputs = np.zeros(len(tt_df), dtype=float)   # the output from the network for a single sequence
        # sum network outputs from all chunks
        for s in range(0, len(seq_chunks), args.n_chunks):
            e = s + args.n_chunks
            outputs += nn_sess.run(None, {'input': seq_chunks[s:e]})[0].sum(axis=0)
        # divide by the number of seq_chunks we processed to get a mean output
        outputs /= len(seq_chunks)

        aggregated.append(outputs)


    # aggregate everything we just pulled from the fasta file
    aggregated = np.array(aggregated)

    # get prediction and maximum probabilities for confidence scoring
    logger.info('getting max probabilities')
    preds = np.argmax(aggregated, axis=1)
    all_maxprobs.append(np.sort(np.partition(aggregated, k)[:, k:])[::-1])
    all_preds.append(preds)

    # build input matrix for confidence model
    all_lengths = np.array(all_lengths, dtype=np.float32)
    all_maxprobs = np.concatenate(all_maxprobs)
    conf_input = np.concatenate([all_lengths[:, np.newaxis], all_maxprobs], axis=1, dtype=np.float32)

    # get confidence probabilities
    logger.info('calculating confidence probabilities')
    conf = conf_sess.run(None, {'float_input': conf_input})[1][:, 1]

    # build the final output data frame
    logger.info('building final output data frame')
    all_preds = np.concatenate(all_preds)
    output = tt_df.iloc[all_preds].copy()
    output['filename'] = all_filepaths
    output['ID'] = all_seqnames
    output['score'] = conf
    output = output[outcols]

    # write out data
    if args.output is None:
        outf = sys.stdout
    else:
        outf = open(args.output, 'w')
    output.to_csv(outf, index=False)

    after = time()
    logger.info(f'Took {after - before:.1f} seconds')

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(levelname)s-%(message)s')
    logging.info('starting')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta_path', type=str,
                        default=None, help='sequence path')
    parser.add_argument('-t', '--txt_file', type=str,
                        default=None, help='txt file with fasta paths')
    parser.add_argument('-o', '--output', type=str,
                        default=None, help='output destination')
    args = parser.parse_args(argv)

    if args.txt_file:
        with open(args.txt_file, 'r') as f:
            fasta_paths = [path.strip() for path in f]
    else:
        fasta_paths = [args.fasta_path,]

    for fasta_path in fasta_paths:
        get_predictions(fasta_path=fasta_path, output_dest=args.output)

    logger.info('finished')


def run_test(argv=None):
    data_path = get_data_path()
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str,
                        default=None, help='output destination')

    args = parser.parse_args(argv)
    get_predictions(fasta_path=data_path,
                    output_dest=args.output)

