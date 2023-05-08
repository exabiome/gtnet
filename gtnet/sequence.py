import logging
import multiprocessing as mp

import numpy as np
import skbio
from skbio.sequence import DNA

__all__ = ['FastaSequenceEncoder', 'FastaReader']


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
        batches = np.zeros((2 * len(starts), self.window), dtype=np.uint8) + self.padval
        for i, (s, e) in enumerate(zip(starts, ends)):
            l = e - s
            batches[2 * i][:l] = fwd[s:e]
            batches[2 * i + 1][:l] = rev[s:e]
        return batches

    @classmethod
    def get_dna_map(cls, vocab=None):
        '''
        Create data structures for mapping DNA sequence to

        Returns
            vocab:      the DNA vocabulary used for building the data structures
            basemap:    a 128 element array for mapping ASCII character values to encoded values
            rcmap:      an array for mapping between complementary characters of encoded values
        '''
        if vocab is None:
            vocab = ('ACYWSKDVNTGRMHB')
        rcmap = cls.get_revcomp_map(vocab)
        basemap = np.zeros(128, dtype=np.uint8)
        # reverse so we store the lowest for self-complementary codes
        for i, c in reversed(list(enumerate(vocab))):
            basemap[ord(c)] = i
            basemap[ord(c.lower())] = i
        basemap[ord('x')] = basemap[ord('X')] = basemap[ord('n')]
        return vocab, basemap, rcmap

    @classmethod
    def get_revcomp_map(cls, vocab):
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


class FastaReader(mp.Process):

    _done = 0xDEADBEEF

    def __init__(self, encoder, *fastas):
        self.encoder = encoder
        self.fastas = fastas

    def _target(self, q, fastas):
        i = 0
        for fa in fastas:
            logging.debug(f'loading {fa}')
            for seq in skbio.read(fa, format='fasta', constructor=DNA, validate=False):
                batches = self.encoder.encode(seq.values)
                val = (fa, seq.metadata['id'], len(seq), batches)
                q.put(val)
            i += 1
        q.put(self._done)

    def read(self, fastas):
        self._q = mp.Queue()
        self._proc = mp.Process(target=self._target, args=(self._q, fastas))
        self._proc.start()
        return self

    def __iter__(self):
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
