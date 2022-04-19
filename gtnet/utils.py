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
    A class that aggregates information from:
        (1) Deployment package manifest - package information for 
        inference, including model and label paths. Also stores
        name of configuration file and character vocab used to train 
        (2) Model configuration - the parameters used for training

    Attributes
    ----------
    manifest : dict
        package deployment information (i.e. model path etc)
    
    inf_model_path : str
        absolute path of NN model

    conf_model_path : str
        absolute path of confidence model

    model_config : dict
        config parameters used for trained model

    window : int
        window size used to chunk and batch sequence

    step : int
        step size used to chunk and batch sequence

    taxa_df_path : str
        absolute path of label file

    chars : str
        character vocabular used by trained model

    pad_value : int
        value to use when padding sequence

    basemap : numpy.ndarray
        DNA map used for encoding of sequence

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
    Extract manifest file and use that to instantiate a config object
    
    Returns
    -------
    config : GTNetConfig object
        the config object for running GTNet
    '''
    deploy_path = os.path.join(resource_filename(__name__, 'gtnet.deploy/'))
    manifest_path = os.path.join(deploy_path, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    config = GTNetConfig(manifest)
    return config


def get_taxon_pred(output):
    '''
    Takes our model output to determine the most confident prediction

    Parameters
    ----------
    output : numpy.ndarray
        predicted output of model for one sequence

    Returns
    -------
    prediction : numpy.int64
        most confident prediction -- based on highest mean network output value
    '''
    prediction = output.mean(axis=0).argmax()
    return prediction

  
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