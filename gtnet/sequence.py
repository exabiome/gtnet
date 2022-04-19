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
        basemap[ord(c.upper())] = i
        basemap[ord(c.lower())] = i
    basemap[ord('x')] = basemap[ord('X')] = basemap[ord('n')]
    return basemap


class DNAEncoder:
    '''
    Creates an Encoder with the provided character sequence
    
    Converts all (string) characters of DNA to integers
    based on where the character is in our chars array
    
    For example, with 'chars = "ATCG"', 'A' would map
    to '0' and 'T' would map to '2' and so forth
    
    Parameters
    ----------
    chars : str
        A string with the DNA characters to encode.
        The position of the base in the string will
        be the integer it gets encoded to

    '''
    def __init__(self, chars):
        self.chars = chars
        self.basemap = _get_DNA_map(self.chars)

    def encode(self, seq):
        '''
        Encode string characters into associated integers

        Parameters
        ----------
        seq : skbio.sequence._dna.DNA
            Our (string) character sequence read from file

        Returns
        -------
        encoded_sequence : numpy.ndarray
            Encoded sequence from associated basemap

        '''
        charar = seq.values.view(np.uint8)
        encoded_sequence = self.basemap[charar] 
        return encoded_sequence


def batch_sequence(sequence, window, padval, step, encoder):
    '''
    Takes a DNA sequence of string characters returns an
    array with the sequence encoded into integers and batched
    in the same manner as the model was trained on
    
    Parameters
    ----------
    sequence : skbio.sequence._dna.DNA
        The character DNA sequence, read from file 
    window : int
        The window size used for batching our sequence
    padval : int
        Which value should be used for padding
    step : int
        Step size used for batching our sequence
    encoder : DNAEncoder
        An instantiation of our DNAEncoder class

    Returns
    -------
    batches : numpy.ndarray
        original sequence broken down into batches
    '''
    encoded_seq = encoder.encode(sequence)
    starts = np.arange(0, encoded_seq.shape[0], step)
    ends = np.minimum(starts + window, encoded_seq.shape[0])
    batches = np.ones((len(starts), window), dtype=int) * padval
    for idx, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        length = end_idx - start_idx
        batches[idx][:length] = encoded_seq[start_idx:end_idx]
    return batches