from gtnet import predict
from gtnet import generate_report
import argparse




parser = argparse.ArgumentParser(...)
parser.add_argument("sequences", dtype=str, help='Fasta file of sequences to predict')

args = parser.parse_args()


kwargs = vars(args)

probas = predict(**kwargs)


report = generate_report(probas, ...)  # pandas.DataFrame


report.to_csv(args.output)
