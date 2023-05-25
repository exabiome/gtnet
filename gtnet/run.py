import argparse
import json
import logging
import os
import pickle
import sys
from time import time

import numpy as np
import pandas as pd
from pkg_resources import resource_filename
import torch
import torch.nn as nn

from .sequence import FastaReader, FastaSequenceEncoder
from .utils import get_logger, DeployPkg


def _load_deploy_pkg():
    pkg = DeployPkg()

    tmp_conf_model = dict()
    for lvl_dat in pkg['conf_model']:
        lvl_dat['taxa'] = np.array(lvl_dat['taxa'])

        lvl_dat['model'] = torch.jit.load(pkg.path(lvl_dat.pop('model')))

        tmp_conf_model[lvl_dat['level']] = lvl_dat

    pkg['conf_model'] = tmp_conf_model

    pkg['vocabulary'] = "".join(pkg['vocabulary'])

    pkg['nn_model'] = torch.jit.load(pkg.path(pkg['nn_model']))

    return pkg['nn_model'], pkg['conf_model'], pkg['training_config'], pkg['vocabulary']


class GPUModel(nn.Module):

    def __init__(self, model, device):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x).cpu()


def run_torchscript_inference(argv=None):
    """
    Convert a Torch model checkpoint to ONNX format
    """
    desc = "Run predictions using ONNX"
    epi = ("")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('fasta', nargs='?', type=str, help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-c', '--n_chunks', type=int, default=10000, help='the number of sequence chunks to process at a time')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output file to save classifications to')
    if torch.cuda.is_available():
        parser.add_argument('-g', '--gpu', action='store_true', default=False, help='Use GPU')
        parser.add_argument('-D', '--device_id', type=int, default=0, choices=torch.arange(torch.cuda.device_count()).tolist(),
                            help='the device ID of the GPU to use')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print specific information about sequences')

    args = parser.parse_args(argv)

    before = time()

    logger = get_logger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    use_gpu = getattr(args, 'gpu', False)
    if use_gpu:
        device = torch.device(args.device_id)
    else:
        device = torch.device('cpu')

    model, conf_models, train_conf, vocab = _load_deploy_pkg()

    window = train_conf['window']
    step = train_conf['step']

    encoder = FastaSequenceEncoder(window, step, vocab=vocab, device=device)
    reader = FastaReader(encoder, args.fasta)

    model = model.to(device)

    output_size = sum(len(lvl['taxa']) for lvl in conf_models.values())

    seqnames = list()
    lengths = list()
    filepaths = list()
    aggregated = list()

    torch.set_grad_enabled(False)

    logger.info(f'Calculating classifications for all sequences in {args.fasta}')
    for file_path, seq_name, seq_len, seq_chunks in reader:
        seqnames.append(seq_name)
        lengths.append(seq_len)
        filepaths.append(file_path)

        logger.debug(f'Getting predictions for windows of {seqnames[-1]}, {seq_chunks.shape[1] * 2} chunks, {lengths[-1]} bases')
        outputs = torch.zeros(output_size, device=device)   # the output from the network for a single sequence
        # sum network outputs from all chunks
        for s in range(0, len(seq_chunks), args.n_chunks):
            e = s + args.n_chunks
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

    # write out data
    if args.output is None:
        outf = sys.stdout
    else:
        outf = open(args.output, 'w')
    output.to_csv(outf, index=True)

    after = time()
    logger.info(f'Took {after - before:.1f} seconds')
