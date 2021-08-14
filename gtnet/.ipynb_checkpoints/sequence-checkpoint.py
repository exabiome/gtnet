import numpy as np
import skbio

__all__ = ['_get_DNA_map',
           'get_sequences',
          'chunk_seq',
           'map_seq',
           'get_rev_comp',
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
    for i, c in reversed(list(enumerate(chars))):  # reverse so we store the lowest for self-complementary codes
        basemap[ord(c)] = i
        basemap[ord(c.lower())] = i
    return chars, basemap

#this will just pull all the sequences from a fasta file
def get_sequences(path):
    #return [seq.values.view(np.uint8) for seq in skbio.io.read(path, format='fasta')]
    return [seq.values for seq in skbio.io.read(path, format='fasta')]

#this will produce a list featuring chunks of the sequence
def chunk_seq(sequence, size):
    num_windows = len(sequence)//size
    return [sequence[size*i: size*(i+1)] for i in range(num_windows)]

#this will map each item onto our basemap from above
def map_seq(chunked_sequence, basemap):
    #return basemap[np.array(chunked_sequence)]#.view(np.uint8)]
    return basemap[np.vectorize(ord)(np.array(chunked_sequence))]

#this will return the reverse complementary strand    
def get_rev_comp(chunks, chars):
    chars_dict = {i:v for i,v in enumerate(chars)}
    return np.vectorize(chars_dict.get)((chunks + 9) %18)

#combines fxns above into a single array with chunks in both directons
def get_bidir_seq(sequence, size=4096):
    chars, basemap = _get_DNA_map()
    chunks = chunk_seq(sequence, size)
    fwd_chunk_mapped = map_seq(chunks, basemap).astype(int)
    rev_chunked = get_rev_comp(fwd_chunk_mapped, chars)
    rev_chunk_mapped = map_seq(rev_chunked, basemap).astype(int)
    bi_seq = np.append(fwd_chunk_mapped, rev_chunk_mapped, axis=0)
    return bi_seq