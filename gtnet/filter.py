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

    tmp_roc = dict()
    for lvl_dat in pkg['conf_model']:
        tmp_roc[lvl_dat['level']] = np.load(pkg.path(lvl_dat['roc']))

    return tmp_roc


class GPUModel(nn.Module):

    def __init__(self, model, device):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x).cpu()


def filter_classifications(argv=None):
    """
    Convert a Torch model checkpoint to ONNX format
    """
    desc = "Run predictions using ONNX"
    epi = ("")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('csv', nargs='?', type=str, help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-f', '--fpr', default=0.05, type=float, help='the false-positive rate to classify to')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output file to save classifications to')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print specific information about sequences')

    args = parser.parse_args(argv)

    before = time()

    if args.csv is None:
        args.csv = sys.stdin

    logger = get_logger()

    rocs = _load_deploy_pkg()

    cutoffs = dict()
    for lvl in rocs:
        roc = rocs[lvl]
        idx = np.searchsorted(roc['fpr'], args.fpr)
        if idx == 0:
            cutoffs[lvl] = 1.0
        else:
            cutoffs[lvl] = roc['thresh'][idx-1]

    levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    df = pd.read_csv(args.csv, index_col='ID')
    mask = np.ones(len(df), dtype=bool)
    data = dict()
    for lvl in levels:
        probs = df[f'{lvl}_prob']
        taxa = df[lvl].copy()
        mask = mask & (probs > cutoffs[lvl])
        taxa[~mask] = None
        data[lvl] = taxa
    output = pd.DataFrame(data)

    # write out data
    if args.output is None:
        outf = sys.stdout
    else:
        outf = open(args.output, 'w')
    output.to_csv(outf, index=True)

    after = time()
    logger.info(f'Took {after - before:.1f} seconds')
