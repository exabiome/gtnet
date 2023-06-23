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
    desc = "Get filtered taxonomic classification for each sequence in a Fasta file."
    epi = ("This command will output a taxonomic classification filtered to a specified false-positive rate "
           "for each sequence")

    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument('fastas', type=str, nargs='+', help='the Fasta files to do taxonomic classification on')
    parser.add_argument('-s', '--seqs', action='store_true', help='provide classification for sequences')
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

    model, conf_models, train_conf, vocab, rocs = load_deploy_pkg(for_predict=True, for_filter=True, contigs=args.seqs)

    window = train_conf['window']
    step = train_conf['step']

    logger.info(f'Getting class predictions for each contig in {",".join(args.fastas)}')
    output = run_torchscript_inference(args.fastas, model, conf_models, window, step, vocab, seqs=args.seqs,
                                      device=device, logger=logger)

    logger.info(f'Getting probability cutoffs for target false-positive rate of {args.fpr}')
    cutoffs = get_cutoffs(rocs, args.fpr)

    logger.info('Filtering class predictions')
    output = filter_predictions(output, cutoffs)

    write_csv(output, args)

    after = time()
    logger.info(f'Took {after - before:.1f} seconds')
