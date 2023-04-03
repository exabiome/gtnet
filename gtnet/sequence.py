import logging
import multiprocessing as mp

import numpy as np
<<<<<<< HEAD
import skbio
from skbio.sequence import DNA

__all__ = ['_get_DNA_map',
           'get_sequences',
           'chunk_seq',
           'get_rev_seq',
           'get_bidir_seq',
           'get_revcomp_map']


def get_revcomp_map(vocab):
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


class FastaSequenceEncoder:

    def __init__(self, window, step, vocab=None, padval=None):
        self.window = window
        self.step = step
        self.vocab, self.basemap, self.rcmap = self.get_dna_map(vocab=vocab)
        if padval is None:
            self.padval = next(i for i in range(len(self.vocab)) if self.vocab[i].upper() == 'N')
        else:
            self.padval = padval

    def encode(self, seq):
        if seq.dtype == np.dtype('S1'):
            seq = seq.view(np.uint8)
        elif seq.dtype == np.dtype('U1'):
            seq = seq.astype('S').view(np.uint8)
        elif seq.dtype != np.uint8:
            raise ValueError('seq must be bytes or uint8')
        fwd = self.basemap[seq]
        rev = self.rcmap[fwd[::-1]]
        starts = np.arange(0, fwd.shape[0], self.step)
        ends = np.minimum(starts + self.window, fwd.shape[0])
        batches = np.zeros((2 * len(starts), self.window), dtype=int) + self.padval
        for i, (s, e) in enumerate(zip(starts, ends)):
            l = e - s
            batches[2 * i][:l] = fwd[s:e]
            batches[2 * i + 1][:l] = rev[s:e]
        return batches

    @staticmethod
    def get_dna_map(vocab=None):
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



class FastaReader(mp.Process):

    _done = 0xDEADBEEF

    def __init__(self, encoder, *fastas):
        self.encoder = encoder
        self.fastas = fastas

    def _target(self, q, fastas):
        i = 0
        for fa in fastas:
            print("Reading sequence", i)
            logging.info(f'loading {fa}')
            for seq in skbio.read(fa, format='fasta', constructor=DNA, validate=False):
                batches = self.encoder.encode(seq.values)
                val = (fa, seq.metadata['id'], len(seq), batches)
                q.put(val)
            i += 1
        q.put(self._done)

    def read(self, fastas):
        print("creating Queue")
        self._q = mp.Queue()
        self._proc = mp.Process(target=self._target, args=(self._q, fastas))
        print("starting Process")
        self._proc.start()
        return self

    def __iter__(self):
        print("returning self")
        return self

    def __next__(self):
        seq = None
        while True:
            seq = self._q.get()
            if seq == self._done:
                self._proc.join()
                raise StopIteration
            break
        return seq


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

    For example, with `chars = "ATCG"`, `A` would map
    to `0` and `T` would map to `2` and so forth

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
            A scikit-bio *DNA* object

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
