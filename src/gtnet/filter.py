import argparse
import sys
from time import time

import numpy as np
import pandas as pd

from .utils import get_logger, load_deploy_pkg, write_csv


def get_cutoffs(rocs, fpr):
    """Get score cutoffs to achieve desired false-positive rate

    Parameters
    ----------

    rocs : dict
        The ROC curves for each taxonomic level

    fpr : float
        The false-positive rate to get the score for
    """
    cutoffs = dict()
    for lvl in rocs:
        roc = rocs[lvl]
        idx = np.searchsorted(roc['fpr'], fpr)
        if idx == 0:
            cutoffs[lvl] = 1.0
        else:
            cutoffs[lvl] = roc['thresh'][idx-1]
    return cutoffs


def filter(argv=None):
    """Filter raw taxonomic classifications
    """
    desc = "Filter raw taxonomic classifications"
    epi = ("")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('csv', nargs='?', type=str, help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-f', '--fpr', default=0.05, type=float, help='the false-positive rate to classify to')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output file to save classifications to')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='print specific information about sequences')

    args = parser.parse_args(argv)

    before = time()

    if args.csv is None:
        args.csv = sys.stdin

    logger = get_logger()

    df = pd.read_csv(args.csv)

    if 'ID' in df.columns:
        seqs = True
        df = df.set_index(['file', 'ID'])
    else:
        df = df.set_index('file')
        seqs = False

    rocs = load_deploy_pkg(for_filter=True, contigs=seqs)

    cutoffs = get_cutoffs(rocs, args.fpr)

    output = filter_predictions(df, cutoffs)
    write_csv(output, args)

    after = time()
    logger.info(f'Took {after - before:.1f} seconds')


def filter_predictions(pred_df, cutoffs):
    """Filter taxonomic classification predictions

    Parameters
    ----------

    pred_df : DataFrame
        The DataFrame containing predictions and confidence scores for each taxonomic level

    cutoffs : dict
        A dictionary containing the confidence score cutoff for each taxonomic level
    """
    levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    mask = np.ones(len(pred_df), dtype=bool)
    data = dict()
    for lvl in levels:
        probs = pred_df[f'{lvl}_prob']
        taxa = pred_df[lvl].copy()
        mask = mask & (probs > cutoffs[lvl])
        taxa[~mask] = None
        data[lvl] = taxa
    return pd.DataFrame(data)
