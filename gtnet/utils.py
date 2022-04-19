from pkg_resources import resource_filename
from .sequence import _get_DNA_map
import pandas as pd
import ruamel.yaml as yaml
import json
import logging
import sys
import os


class GTNetConfig:
    '''
    class that aggregates information from:
        (1) Deployment package manifest
            + determines/stores absolute paths
        (2) Model configuration
    '''
    def __init__(self, manifest):
        self.manifest = manifest
        self.inf_model_path = self._get_abs_path(self.manifest['nn_model'])
        self.conf_model_path = self._get_abs_path(self.manifest['conf_model']) 
        self.model_config = self._get_model_config()
        self.window = self.model_config['window']
        self.step = self.model_config['step']
        self.taxa_df_path = self._get_abs_path(self.manifest['taxa_table'])
        self.chars = ''.join(self.manifest['vocabulary'])
        self.pad_value = self.chars.find('N')
        self.basemap = _get_DNA_map(self.chars)
        
    def _get_abs_path(self, relative_path):
        return resource_filename(__name__, relative_path)

    def _get_model_config(self):
        config_path = self._get_abs_path(self.manifest['training_config'])
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        return model_config


def get_config():
    '''
    Extract manifest file and use that to instantiate our config class
    '''
    deploy_path = os.path.join(resource_filename(__name__, 'gtnet.deploy/'))
    manifest_path = os.path.join(deploy_path, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    config = GTNetConfig(manifest)
    return config


def get_taxon_pred(output):
    return output.mean(axis=0).argmax()

  
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