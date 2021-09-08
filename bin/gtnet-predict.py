from gtnet.run import predict
import argparse
import logging


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(levelname)s-%(message)s')
    logging.info('starting')

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta_path', type=str,
                        default=None, help='sequence path')
    parser.add_argument('-d', '--domain', type=str,
                        default=None, help='domain',
                        choices=['bacteria', 'archaea'])
    parser.add_argument('-v', '--vocab', type=str,
                        default=None, help='vocabulary')
    args = parser.parse_args()

    predict(fasta_path=args.fasta_path, domain=args.domain,
            vocab=args.vocab)

    logging.info('finished!')


if __name__ == '__main__':
    main()
