import numpy as np

__all__ = [
           '_get_DNA_map',
           'DNA_map_encoder'
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


class DNA_map_encoder:
    '''
    stores our characters and encodes sequences
    '''
    chars, basemap = _get_DNA_map()
    
    @classmethod
    def characters(cls):
        return cls.chars

    @classmethod
    def encode(cls, seq):
        charar = seq.values.view(np.uint8)
        return cls.basemap[charar]


#def function_for_windowing():