import os
from pkg_resources import resource_filename
import json
import logging
import sys


def get_species_pred(output):
    pass


def get_label_file():
    name_path = os.path.join(resource_filename(__name__, 'gtnet.deploy/taxa_table.csv'))
    with open(name_path) as taxon_file:
         taxon_df = pd.read_csv(taxon_file) 
    return taxon_df

  
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