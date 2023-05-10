import logging
import sys

import numpy as np
import skbio
from skbio.sequence import DNA
import torch
import torch.nn.functional as F


import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

__all__ = ['FastaSequenceEncoder', 'FastaReader']


class FastaSequenceEncoder:

    def __init__(self, window, step, vocab=None, padval=None, min_seq_len=100, device=torch.device('cpu')):
        self.window = window
        self.step = step
        self.vocab, self.basemap, self.rcmap = self.get_dna_map(vocab=vocab)
        if padval is None:
            self.padval = next(i for i in range(len(self.vocab)) if self.vocab[i].upper() == 'N')
        else:
            self.padval = padval

        self.padval = self.padval
        self.min_seq_len = min_seq_len
        self.device = device

    def encode(self, seq):
        if seq.dtype == np.dtype('S1'):
            seq = seq.view(np.uint8)
        elif seq.dtype == np.dtype('U1'):
            seq = seq.astype('S').view(np.uint8)
        elif seq.dtype != np.uint8:
            raise ValueError('seq must be bytes or uint8')

        fwd = self.basemap[seq]
        rev = self.rcmap[fwd[::-1]]
        fwd = torch.from_numpy(fwd)
        rev = torch.from_numpy(rev)

        C_min = len(seq) % self.step
        if C_min == 0:
            C_min = self.step
        n_short_C = max(((self.min_seq_len - C_min - 1) // self.step) + 1, 0)
        n_C = (len(seq) - 1) // self.step + 1
        n_chunks = n_C - n_short_C
        padlen =  (n_chunks - 1) * self.step + self.window - len(seq)

        fwd = F.pad(fwd, (0, padlen), "constant", self.padval)
        rev = F.pad(rev, (0, padlen), "constant", self.padval)

        ret = torch.row_stack([fwd, rev]).to(self.device)
        ret = ret.unfold(1, self.window, self.step)

        return ret

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
        rcmap = np.zeros(len(vocab), dtype=np.uint8)
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

    @classmethod
    def _target(cls, q, fastas, encoder, finished):
        i = 0
        for fa in fastas:
            logging.debug(f'loading {fa}')
            for seq in skbio.read(fa, format='fasta', constructor=DNA, validate=False):
                batches = encoder.encode(seq.values)
                val = (fa, seq.metadata['id'], len(seq), batches)
                result = q.put(val)
            i += 1
        q.put(cls._done)
        finished.wait()

    def __iter__(self):
        self._q = mp.Queue()
        self._finished = mp.Event()
        self._proc = mp.Process(target=self._target, args=(self._q, self.fastas, self.encoder, self._finished))
        self._proc.start()
        return self

    def __next__(self):
        seq = None
        while True:
            seq = self._q.get()
            if seq == self._done:
                self._finished.set()
                self._proc.join()
                raise StopIteration
            break
        return seq
