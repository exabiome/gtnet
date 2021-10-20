import os
from pkg_resources import resource_filename
import json
import logging
import sys


def get_species_pred(output):
    return output.mean(axis=0).argmax()


def get_label_file(domain='archaea'):
    if(domain == 'archaea'):
        path = 'arc_names.json'
    else:
        path = 'bac_names.json'

    name_path = os.path.join(resource_filename(__name__, 'labels'), path)
    with open(name_path) as species_file:
        species_names = json.load(species_file)
    return species_names

  
def parse_logger(string):
    if not string:
        ret = logging.getLogger('stdout')
        hdlr = logging.StreamHandler(sys.stderr)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)
    ret.setLevel(logging.INFO)
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return ret


def get_logger():
    return parse_logger('')



def get_data_path():
    """
    Get the path to the test data 
    Returns -- path: str (absolute path to test data file)
    """
    file_name = 'GCA_000006155.2.fna'
    return os.path.join(resource_filename(__name__, 'data'), file_name)