from model import load_model
from sequence import _get_DNA_map, get_sequences, get_bidir_seq
import numpy as np
import argparse
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


def run_onnx_inference(argv=None):
    """
    Convert a Torch model checkpoint to ONNX format
    """

    import argparse
    import json
    from time import time

    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view as swv
    import onnxruntime as ort
    import pandas as pd
    import ruamel.yaml as yaml
    import skbio
    from skbio.sequence import DNA

    from deep_taxon.sequence.convert import DNAVocabIterator
    from deep_taxon.utils import get_logger


    desc = "Run predictions using ONNX"
    epi = ("")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('deploy_dir', type=str, help='the directory containing all the data for deployment')
    parser.add_argument('fastas', nargs='+', type=str, help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-c', '--n_chunks', type=int, default=10000, help='the number of sequence chunks to process at a time')
    parser.add_argument('-F', '--fof', action='store_true', default=False, help='a file-of-files was passed in')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output file to save classifications to')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print specific information about sequences')

    args = parser.parse_args(argv)

    if args.fof:
        with open(args.fastas[0], 'r') as f:
            args.fastas = [s.strip() for s in f.readlines()]

    files = ('taxa_table', 'nn_model', 'conf_model', 'training_config')
    # read manifest
    with open(os.path.join(args.deploy_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    # remap files in deploy_dir to be relative to where we are running
    for key in files:
        manifest[key] = os.path.join(args.deploy_dir, os.path.basename(manifest[key]))

    model_path = manifest['nn_model']
    conf_model_path = manifest['conf_model']
    config_path = manifest['training_config']
    tt_path = manifest['taxa_table']
    vocab = "".join(manifest['vocabulary'])

    logger = get_logger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    logger.info(f'loading model from {model_path}')
    nn_sess = ort.InferenceSession(model_path, sess_options=so)

    logger.info(f'loading confidence model from {conf_model_path}')
    conf_sess = ort.InferenceSession(conf_model_path, sess_options=so)
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

    logger.info(f'beginning inference')
    before = time()

    encoder = FastaSequenceEncoder(window, step, vocab=vocab)
    reader = FastaReader(encoder)

    n_seqs = 0
    all_seqnames = list()
    all_lengths = list()
    for file_path, seq_name, seq_len, seq_chunks in reader.read(args.fasta):
        all_seqnames.append(seq_name)
        all_lengths.append(seq_len)
        all_filepaths.extend(file_path)

        logging.debug(f'getting outputs for {all_seqnames[-1]}, {len(batches)} chunks, {all_lengths[-1]} bases')
        outputs = np.zeros(len(tt_df), dtype=float)
        aggregated = list()
        for s in range(0, len(batches), args.n_chunks):
            e = s + args.n_chunks
            outputs += nn_sess.run(None, {'input': batches[s:e]})[0].sum(axis=0)
        outputs /= len(batches)
        aggregated.append(outputs)


    for fasta_path in args.fastas:
        preds = list()

        for seq_name, len, batches in reader.readseq(fasta_path):

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
    parser.add_argument('-m', '--model_path', type=str,
                        default=None, help='path to onnx model')
    parser.add_argument('-v', '--vocab', type=str,
                        default=None, help='vocabulary')
    args = parser.parse_args()
    predict(fasta_path=args.fasta_path, model_path=args.model_path,
            vocab=args.vocab)
    logging.info('finished!')


if __name__ == '__main__':
    main()
