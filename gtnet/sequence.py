import numpy as np
import skbio


__all__ = ['_get_DNA_map',
           'get_sequences',
           'chunk_seq',
           'get_rev_seq',
           'get_bidir_seq',
           'get_revcomp_map']


def get_revcomp_map(vocab):
    '''
    Based on desired vocabulary, will return a numpy array
    to be used for reverse complement mapping -- default vocab
    can be found in _get_DNA_map function below
    '''
    chars = {
        'A': 'T',
        'G': 'C',
        'C': 'G',
        'T': 'A',
        'Y': 'R',
        'R': 'Y',
        'W': 'W',
        'S': 'S',
        'K': 'M',
        'M': 'K',
        'D': 'H',
        'V': 'B',
        'H': 'D',
        'B': 'V',
        'X': 'X',
        'N': 'N',
    }

    d = {c: i for i, c in enumerate(vocab)}
    rcmap = np.zeros(len(vocab), dtype=int)
    for i, base in enumerate(vocab):
        rc_base = chars[base]
        base_i = d[base]
        rc_base_i = d[rc_base]
        rcmap[base_i] = rc_base_i
    return rcmap


def _get_DNA_map(vocab=None):
    '''
    create a DNA map with some redundancy so that we can
    do base-complements with +/% operations.
    Using this scheme, the complement of a base should be:
    (base_integer + 9) % 18
    chars[(basemap[ord('N')]+9)%18]
    For this to work, bases need to be ordered as they are below
    '''
    if vocab is None:
        vocab = ('ACYWSKDVNTGRMHB')
    rcmap = get_revcomp_map(vocab)
    basemap = np.zeros(128, dtype=np.uint8)
    # reverse so we store the lowest for self-complementary codes
    for i, c in reversed(list(enumerate(vocab))):
        basemap[ord(c)] = i
        basemap[ord(c.lower())] = i
    basemap[ord('x')] = basemap[ord('X')] = basemap[ord('n')]
    return vocab, basemap, rcmap


def get_sequences(path, basemap):
    """
    (1) pull all the sequences from a fasta file with scikit-bio
    (2) return a list of the sequences mapped to 8 bit integers via
    the provided basemap -- each item in list is a numpy array
    * this will return a jagged array *
    """
    seqs = [seq.values for seq in skbio.io.read(path, format='fasta')]
    return [basemap[seq.view(np.int8)] for seq in seqs]


def chunk_seq(seq, chunk_size, pad_value):
    """
    (1) determine the padding length required from provided chunk size
    (2) pad sequence on right side only with specified padding value
    (3) reshape the sequence to have rows with length of chunk size
    *only padding once on full sequence -- this ensures entire sequence
    is divisible by the chunk size*
    """
    padding_len = chunk_size - len(seq) % chunk_size
    seq = np.pad(seq, (0, padding_len), constant_values=(0, pad_value))
    return seq.reshape((-1, chunk_size))


def get_rev_seq(seq, rcmap):
    """
    (1) flips the sequence and does a index lookup in one operation
    with our reverse complement map
    """
    return rcmap[np.flip(seq)]


def get_bidir_seq(fwd_seq, rcmap, chunk_size, pad_value):
    """
    (1) determines the reverse complement of fwd sequence
    (2) chunk both forward and reverse sequences
    (3) append fwd and reverse sequence into one array
    """
    rev_seq = get_rev_seq(fwd_seq, rcmap)
    fwd_chunks = chunk_seq(fwd_seq, chunk_size, pad_value)
    rev_chunks = chunk_seq(rev_seq, chunk_size, pad_value)
    return np.append(fwd_chunks, rev_chunks, axis=0)
