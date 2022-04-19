import numpy as np


def _get_DNA_map(chars):
    '''
    create a DNA map with some redundancy so that we can
    do base-complements with +/% operations.
    Using this scheme, the complement of a base should be:
    (base_integer + 9) % 18
    chars[(basemap[ord('N')]+9)%18]
    For this to work, bases need to be ordered as they are below
    '''
    basemap = np.zeros(128, dtype=np.uint8)
    for i, c in reversed(list(enumerate(chars))):  # reverse so we store the lowest for self-complementary codes
        basemap[ord(c)] = i
        basemap[ord(c.lower())] = i
    basemap[ord('x')] = basemap[ord('X')] = basemap[ord('n')]
    return basemap


class DNAEncoder:
    '''
    encodes sequences
    '''
    def __init__(self, chars):
        self.chars = chars
        self.basemap = _get_DNA_map(self.chars)

    def encode(self, seq):
        charar = seq.values.view(np.uint8)
        return self.basemap[charar]


def batch_sequence(sequence, window, padval, step, encoder):
    '''
    (0) encode sequence
    (1) determine start + end points
    (2) create empty batches matrix of appropriate size
    *initialize with pad value*
    (3) populate matrix with encoded sequence information
    '''
    encoded_seq = encoder.encode(sequence)
    starts = np.arange(0, encoded_seq.shape[0], step)
    ends = np.minimum(starts + window, encoded_seq.shape[0])
    batches = np.ones((len(starts), window), dtype=int) * padval
    for idx, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        length = end_idx - start_idx
        batches[idx][:length] = encoded_seq[start_idx:end_idx]
    return batches