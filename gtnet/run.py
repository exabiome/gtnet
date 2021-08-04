from .model import load_model
from .sequence import chunk_sequences

def predict(sequences=None, **kwargs):
    if fasta_path is None:
        raise ValueError('sequences must not be None')

    model = load_model()

    for seq in chunk_sequences(fasta_path):
        # 1. chunk sequences
        # 2. pass chunks into model
        # 3. softmax outputs
        # 4. return probabilities
