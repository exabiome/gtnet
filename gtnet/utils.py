from pkg_resources import resource_filename
from .sequence import _get_DNA_map
import pandas as pd
import ruamel.yaml as yaml
import json
import logging
import sys
import os


class gtnet_config:
    def __init__(self, manifest, model_config):
        self.manifest = manifest
        self.model_config = model_config
        self.chars = ''.join(self.manifest['vocabulary'])
        self.pad_value = self.chars.find('N')
        self.basemap = _get_DNA_map(self.chars)
        self.window = self.model_config['window']
        self.step = self.model_config['step']
        self.inf_model_path = self._get_model_path(self.manifest['nn_model'])
        self.conf_model_path = self._get_model_path(self.manifest['conf_model']) 
    def _get_model_path(self, model_path):
        return resource_filename(__name__, model_path)


def get_config():
    deploy_path = os.path.join(resource_filename(__name__, 'gtnet.deploy/'))
    config_path = os.path.join(deploy_path, 'config.yml')
    manifest_path = os.path.join(deploy_path, 'manifest.json')
    with open(config_path, 'r') as  f:
        model_config = yaml.safe_load(f)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    config = gtnet_config(manifest, model_config)
    return config


def get_taxon_pred(output):
    return output.mean(axis=0).argmax()


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