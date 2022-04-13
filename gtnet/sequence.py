import numpy as np
from abc import ABCMeta, abstractmethod
import re


__all__ = ['_get_DNA_map',
           'AbstractSeqIterator',
           'VocabIterator',
           'DNAVocabIterator'
          ]


def _get_DNA_map():
    '''
    create a DNA map with some redundancy so that we can
    do base-complements with +/% operations.
    Using this scheme, the complement of a base should be:
    (base_integer + 9) % 18
    chars[(basemap[ord('N')]+9)%18]
    For this to work, bases need to be ordered as they are below
    '''
    chars = ('ACYWSKDVNTGRMHB')
    basemap = np.zeros(128, dtype=np.uint8)
    for i, c in reversed(list(enumerate(chars))):  # reverse so we store the lowest for self-complementary codes
        basemap[ord(c)] = i
        basemap[ord(c.lower())] = i
    basemap[ord('x')] = basemap[ord('X')] = basemap[ord('n')]
    return chars, basemap
    
    
class AbstractSeqIterator(object, metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def characters(cls):
        pass

    @abstractmethod
    def pack(self, seq):
        pass

    ltag_re = re.compile('>lcl|([A-Za-z0-9_.]+)')

    def __init__(self, paths, logger=None, faa=False, min_seq_len=None):
        self.logger = logger

        # setup our characters
        chars = self.characters()
        chars = list(chars.upper() + chars.lower())
        self.nchars = len(chars)//2

        self.ohe = OneHotEncoder(sparse=False)
        self.ohe.fit(np.array(chars).reshape(-1, 1))

        if isinstance(paths, str):
            paths = [paths]

        self.__paths = paths
        self.__path_iter = None
        self.__name_queue = None
        self.__len_queue = None
        self.__index_queue = None
        self.__taxon_queue = None
        self.__id_queue = None
        self.__total_len = 0
        self.__nseqs = 0
        self.skbio_cls = Protein if faa else DNA
        self.logger.info('reading %s' % self.skbio_cls.__name__)

        self.__curr_block = np.zeros((0, self.nchars), dtype=np.uint8)
        self.__curr_block_idx = 0

        if min_seq_len is None:
            if faa:
                min_seq_len = 150
            else:
                min_seq_len = 50
        self.min_seq_len = min_seq_len

    @property
    def names(self):
        return self.__name_queue

    @property
    def taxon(self):
        return self.__taxon_queue

    @property
    def index(self):
        return self.__index_queue

    @property
    def seqlens(self):
        return self.__len_queue

    @property
    def ids(self):
        return self.__id_queue

    @property
    def index_iter(self):
        return QueueIterator(self.index, self)

    @property
    def names_iter(self):
        return QueueIterator(self.names, self)

    @property
    def id_iter(self):
        return QueueIterator(self.ids, self)

    @property
    def seqlens_iter(self):
        return QueueIterator(self.seqlens, self)

    @property
    def taxon_iter(self):
        return QueueIterator(self.taxon, self)

    def __iter__(self):
        self.__path_iter = iter(self.__paths)
        # initialize the sequence iterator
        self.__curr_iter = self._read_seq(next(self.__path_iter))
        self.__name_queue = deque()
        self.__index_queue = deque()
        self.__len_queue = deque()
        self.__taxon_queue = deque()
        self.__id_queue = deque()
        self.__total_len = 0
        self.__nseqs = 0
        self.__curr_block = np.zeros((0, self.nchars), dtype=np.uint8)
        self.__curr_block_idx = 0
        self.__curr_file_idx = 0
        return self

    @property
    def total_len(self):
        return self.__total_len

    @property
    def curr_block_idx(self):
        return self.__curr_block_idx

    @property
    def curr_block(self):
        return self.__curr_block

    def __read_next_seq(self):
        while True:
            try:
                seq, seqname = next(self.__curr_iter)
                if len(seq) <= self.min_seq_len:
                    continue
                self.__name_queue.append(seqname)
                self.__len_queue.append(len(seq))
                self.__taxon_queue.append(np.uint16(self.__curr_file_idx))
                self.__id_queue.append(self.__nseqs)
                self.__nseqs += 1
                return seq
            except StopIteration:
                try:
                    self.__curr_iter = self._read_seq(next(self.__path_iter))
                    self.__curr_file_idx += 1
                except StopIteration:
                    self.__name_queue.append(None)
                    self.__index_queue.append(None)
                    self.__taxon_queue.append(None)
                    self.__name_queue.append(None)
                    raise StopIteration()

    def __next__(self):
        self._load_buffer()
        ret = self.__curr_block[self.__curr_block_idx]
        self.__curr_block_idx += 1
        return ret

    def _load_buffer(self):
        if self.__curr_block_idx == self.__curr_block.shape[0]:
            # this raises the final StopIteration
            # when nothing is left to read
            seq = self.__read_next_seq()
            self.__curr_block = self.pack(seq)
            self.__total_len += self.__curr_block.shape[0]
            self.__index_queue.append(self.__total_len)
            self.__curr_block_idx = 0

    @classmethod
    def get_seqname(cls, seq):
        return seq.metadata['id']

    def _read_seq(self, path):
        self.logger.info('reading %s', path)
        kwargs = {'format': 'fasta', 'constructor': self.skbio_cls, 'validate': False}
        for seq_i, seq in enumerate(skbio.io.read(path, **kwargs)):
            ltag = self.get_seqname(seq)
            yield seq, ltag

class VocabIterator(AbstractSeqIterator):

    chars, basemap = None, None

    @classmethod
    def characters(cls):
        return cls.chars

    @classmethod
    def encode(cls, seq):
        charar = seq.values.view(np.uint8)
        return cls.basemap[charar]

    def __init__(self, paths, logger=None, min_seq_len=None):
        super().__init__(paths, logger=logger, min_seq_len=min_seq_len)
        self._enc_vocab = np.array(list(self.characters()))

    def pack(self, seq):
        return self.encode(pack)

    @property
    def encoded_vocab(self):
        return self._enc_vocab
    
class DNAVocabIterator(VocabIterator):

    chars, basemap = _get_DNA_map()