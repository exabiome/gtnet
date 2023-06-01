import argparse
import logging
from time import time

from .utils import check_cuda, check_device, get_logger, load_deploy_pkg, write_csv
from .filter import filter_predictions, get_cutoffs
from .predict import run_torchscript_inference


DEFAULT_N_CHUNKS = 10000


def classify(argv=None):
    """
    Get taxonomic classification for each sequence in a Fasta file.

    Parameters
    ----------

    argv : Namespace, default=sys.argv
        The command-line arguments to use for running this command
    """
    desc = "Get taxonomic classification for each sequence in a Fasta file."
    epi = ()

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('fasta', type=str, help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-c', '--n_chunks', type=int, default=DEFAULT_N_CHUNKS,
                        help='the number of sequence chunks to process at a time')
    parser.add_argument('-o', '--output', type=str, default=None, help='the output file to save classifications to')
    check_cuda(parser)
    parser.add_argument('-f', '--fpr', default=0.05, type=float, help='the false-positive rate to classify to')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='print specific information about sequences')

    args = parser.parse_args(argv)

    before = time()

    logger = get_logger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    device = check_device(args)

    model, conf_models, train_conf, vocab, rocs = load_deploy_pkg(for_predict=True, for_filter=True)

    window = train_conf['window']
    step = train_conf['step']

    output = run_torchscript_inference(args.fasta, model, conf_models, window, step, vocab, device=device)

    cutoffs = get_cutoffs(rocs, args.fpr)

    output = filter_predictions(output, cutoffs)

    write_csv(output, args)

    after = time()
    logger.info(f'Took {after - before:.1f} seconds')