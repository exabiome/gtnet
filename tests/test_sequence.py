import tempfile

import numpy as np

from gtnet.sequence import FastaSequenceEncoder, FastaReader

VOCAB = 'ACYWSKDVNTGRMHB'


def get_fasta():
    """TODO - modify this to write to a temp file and return the tempfile path"""
    fl = [">seq1 Sulfolobus",
          "AAGCTNGGGCATTTCAGGGTGAGCCCGGGCAATACAGGG",
          ">seq2 Halobacteria",
          "AAGCCTTGGCAGTGCAGGGTGAGCCGTGGCCGGGCACGGTA",
          ">seq3 Salmonella",
          "ACCGGTTGGCCGTTCAGGGTACAGGTTGGCCGTTCAGGGTAA",
          ">seq4 E.coli",
          "AAACCCTTGCCGTTACGCTTAAACCGAGGCCGGGACAC",
          ">seq5 Y.pestis",
          "AAACCCTTGCCGGTACGCTTAAACCATTGCCGGTACGCTT"]
    _, path = tempfile.mkstemp()
    with open(path, 'w') as f:
        for line in fl:
            print(line, file=f)
    return path


def test_encoder():
    encoder = FastaSequenceEncoder(3, 2, vocab=VOCAB)
    seq = np.array(list("ACTGNGTCNA"))
    batch = encoder.encode(seq)
    expected = np.array([
        [  0,  1,  9],  # ACT
        [  9,  8, 10],  # TNG
        [  9, 10,  8],  # TGN
        [ 10,  0,  1],  # GAC
        [  8, 10,  9],  # NGT
        [  1,  8,  1],  # CNC
        [  9,  1,  8],  # TCN
        [  1,  0, 10],  # CAG
        [  8,  0,  8],  # NA
        [ 10,  9,  8],  # GT
    ])
    np.testing.assert_array_equal(expected, batch)


def test_reader():
    encoder = FastaSequenceEncoder(8, 2, vocab=VOCAB)
    reader = FastaReader(encoder)
    fa = get_fasta()
    i = 0
    for seq in reader.read([fa]):
        assert len(seq) == 4
        i += 1
    assert i == 5

