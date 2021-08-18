import numpy as np
import skbio

__all__ = ['_get_DNA_map',
           'get_sequences',
           'chunk_seq',
           'get_rev_seq',
           'get_bidir_seq']


def _get_DNA_map():
    '''
    create a DNA map with some redundancy so that we can
    do base-complements with +/% operations.
    Using this scheme, the complement of a base should be:
    (base_integer + 9) % 18
    chars[(basemap[ord('N')]+9)%18]
    For this to work, bases need to be ordered as they are below
    '''
    chars = ('ACYWSKDVN'
             'TGRWSMHBN')
    basemap = np.zeros(128, dtype=np.uint8)
    # reverse so we store the lowest for self-complementary codes
    for i, c in reversed(list(enumerate(chars))):
        basemap[ord(c)] = i
        basemap[ord(c.lower())] = i
    return chars, basemap


# this will just pull all the sequences from a fasta file
def get_sequences(path, basemap):
    seqs = [seq.values for seq in skbio.io.read(path, format='fasta')]
    return [basemap[seq.view(np.int8)] for seq in seqs]


# this will pad + chunk sequence and return numpy array
def chunk_seq(seq, size=4096):
    padding_len = size - len(seq) % 4096
    seq = np.pad(seq, (0, padding_len), constant_values=(0, 8))
    return seq.reshape((-1, size))


# this will return the reverse complementary strand
def get_rev_seq(seq, rcmap):
    return rcmap[np.flip(seq)]


# combines fxns above into a single array with chunks in both directons
def get_bidir_seq(fwd_seq, rcmap):
    rev_seq = get_rev_seq(fwd_seq, rcmap)
    fwd_chunks = chunk_seq(fwd_seq, 4096)
    rev_chunks = chunk_seq(rev_seq, 4096)
    return np.append(fwd_chunks, rev_chunks, axis=0)
