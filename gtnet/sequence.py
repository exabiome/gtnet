from itertools import chain
import logging

import numpy as np
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

        if min_seq_len > window:
            raise ValueError("min_seq_len must be less than or equal to window")

        self.padval = self.padval
        self.min_seq_len = min_seq_len
        self.device = device

    def encode(self, seq):
        if seq.dtype == np.dtype('S1'):
            seq = seq.view(np.uint8)
        elif seq.dtype == np.dtype('U1'):
            seq = seq.astype('S').view(np.uint8)
        elif not issubclass(seq.dtype.type, np.integer):
            raise ValueError('seq must be bytes or integer type')

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


class Loader:

    def __init__(self, encoder, *fastas):
        self.encoder = encoder
        self.fastas = fastas

    @staticmethod
    def readfile(fasta_path):
        chars = None
        seqid = None
        with open(fasta_path, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    yield seqid, np.fromiter(chain(*chars), dtype=np.uint8)
                    break
                if line[0] == 62:
                    if seqid is not None:
                        yield seqid, np.fromiter(chain(*chars), dtype=np.uint8)
                    seqid = line[1:line.find(32)].decode('utf8')
                    chars = list()
                else:
                    chars.append(line[:-1])

    @classmethod
    def readfiles(cls, encoder, fastas):
        for fa in fastas:
            logging.debug(f'loading {fa}')
            for seqid, values in cls.readfile(fa):
                batches = encoder.encode(values)
                val = (fa, seqid, len(values), batches)
                yield val


class SerialLoader(Loader):

    def __iter__(self):
        return self.readfiles(self.encoder, self.fastas)


class ParallelLoader(Loader):

    _done = 0xDEADBEEF

    @classmethod
    def _target(cls, q, fastas, encoder, finished):
        for val in cls.readfiles(encoder, fastas):
            q.put(val)
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


class FastaReader(mp.Process):


    def __init__(self, encoder, *fastas, parallel=False):
        if parallel:
            self.loader = ParallelLoader(encoder, *fastas)
        else:
            self.loader = SerialLoader(encoder, *fastas)

    def __iter__(self):
        return iter(self.loader)
