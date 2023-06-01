import argparse
import logging
from time import time

import pandas as pd
import torch

from .sequence import FastaReader, FastaSequenceEncoder
from .utils import check_cuda, check_device, get_logger, load_deploy_pkg, write_csv


DEFAULT_N_CHUNKS = 10000


def predict(argv=None):
    """
    Get network predictions for each sequence in Fasta file

    Parameters
    ----------

    argv : Namespace, default=sys.argv
        The command-line arguments to use for running this command
    """
    desc = "Get network predictions for each sequence in a Fasta file"
    epi = ("This command will output the best classification for every sequence at every taxonomic level, including "
           "confidence scores for each taxonomic level. These classifications can be subsequently filtered using the "
           "'filter' command. See the 'classify' command for getting pre-filtered classifications")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('fasta', type=str, help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-c', '--n_chunks', type=int, default=DEFAULT_N_CHUNKS,
                        help='the number of sequence chunks to process at a time')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output file to save classifications to')
    check_cuda(parser)
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='print specific information about sequences')

    args = parser.parse_args(argv)

    before = time()

    logger = get_logger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    device = check_device(args)

    model, conf_models, train_conf, vocab = load_deploy_pkg(for_predict=True)

    window = train_conf['window']
    step = train_conf['step']

    output = run_torchscript_inference(args.fasta, model, conf_models, window, step, vocab, device=device)

    # write out data
    write_csv(output, args)

    after = time()
    logger.info(f'Took {after - before:.1f} seconds')


def run_torchscript_inference(fasta, model, conf_models, window, step, vocab, n_chunks=DEFAULT_N_CHUNKS,
                              device=torch.device('cpu'), logger=None):
    """Run Torchscript inference

    Parameters
    ----------

    fasta : str
        The path to the Fasta file with sequences to do inference on

    model : RecursiveScriptModule
        The Torchscript model to run inference with

    conf_models : dict
        A dictionary with the confidence model for each taxonomic level. Each model should be a RecursiveScriptModule.
        The expected keys in this dict are 'domain', 'phylum', 'class', 'order', 'family', 'genus' and 'species'.

    window : int
        The length of the sliding window to use for doing inference

    step : int
        The length of the step of the sliding window to use for doing inference

    vocab : str
        The vocabulary used for training `model`

    n_chunks : int, default=10000
        The length of the step of the sliding window to use for doing inference

    device : device, default=torch.device('cpu')
        The Pytorch device to run inference on

    logger : Logger
        The Python logger to use when running inference
    """

    if logger is None:
        logger = get_logger()
        logger.setLevel(logging.CRITICAL)

    encoder = FastaSequenceEncoder(window, step, vocab=vocab, device=device)
    reader = FastaReader(encoder, fasta)

    model = model.to(device)

    output_size = sum(len(lvl['taxa']) for lvl in conf_models.values())

    seqnames = list()
    lengths = list()
    filepaths = list()
    aggregated = list()

    torch.set_grad_enabled(False)

    logger.info(f'Calculating classifications for all sequences in {fasta}')
    for file_path, seq_name, seq_len, seq_chunks in reader:
        seqnames.append(seq_name)
        lengths.append(seq_len)
        filepaths.append(file_path)

        logger.debug((f'Getting predictions for windows of {seqnames[-1]}, '
                      f'{seq_chunks.shape[1] * 2} chunks, {lengths[-1]} bases'))
        outputs = torch.zeros(output_size, device=device)   # the output from the network for a single sequence
        # sum network outputs from all chunks
        for s in range(0, len(seq_chunks), n_chunks):
            e = s + n_chunks
            outputs += model(seq_chunks[0, s:e]).sum(dim=0)
            outputs += model(seq_chunks[1, s:e]).sum(dim=0)
        # divide by the number of seq_chunks we processed to get a mean output
        outputs /= (seq_chunks.shape[1] * 2)
        del seq_chunks

        aggregated.append(outputs)

    lengths = torch.tensor(lengths, device=device)


    # aggregate everything we just pulled from the fasta file
    all_levels_aggregated = torch.row_stack(aggregated)
    del aggregated

    output_data = {'ID': seqnames}

    s = 0
    for lvl, e in zip(model.levels, model.parse):
        conf_model_info = conf_models[lvl]
        taxa = conf_model_info['taxa']
        conf_model = conf_model_info['model'].to(device)

        # determine the number of top k probabilities
        # to use for confidence scoring
        top_k = 2

        logger.info(f'Getting {lvl} predictions for all sequences')
        aggregated = all_levels_aggregated[:, s:e]
        output_data[lvl] = taxa[torch.argmax(aggregated, dim=1).cpu()]

        # get prediction and maximum probabilities for confidence scoring
        logger.debug(f'Getting max {top_k} probabilities for {lvl}')
        maxprobs = torch.topk(aggregated, top_k, dim=1, largest=True, sorted=True).values

        # build input matrix for confidence model
        logger.debug('Calculating confidence probabilities')
        conf_input = torch.column_stack([lengths, maxprobs])

        output_data[f'{lvl}_prob'] = conf_model(conf_input).cpu().numpy().squeeze()

        # set next left bound for all_levels_aggregated
        s = e

    output = pd.DataFrame(output_data).set_index('ID')

    return output
