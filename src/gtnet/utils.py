import glob
import hashlib
import json
import logging
import os
import sys
import urllib
import warnings
import zipfile

import numpy as np
from pkg_resources import resource_filename
import torch
import torch.nn as nn


def parse_logger(string):
    if not string:
        ret = logging.getLogger()
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


class DeployPkg:
    """A class to handle loading and manipulating the deployment package"""

    _deploy_pkg_url = "https://osf.io/download/mwgb9/"

    _checksum = "0245fcf825bfe4de0770fcb46798ca90"

    @classmethod
    def check_pkg(cls):
        deploy_dir = resource_filename(__name__, 'deploy_pkg')
        total = 0
        for path in glob.glob(f"{deploy_dir}/*"):
            total += os.path.getsize(path)
        if total == 0:
            msg = ("Downloading GTNet deployment package. This will only happen on the first invocation "
                   "of gtnet predict or gtnet classify")
            warnings.warn(msg)
            zip_path = resource_filename(__name__, 'deploy_pkg.zip')
            urllib.request.urlretrieve(cls._deploy_pkg_url, zip_path)
            dl_checksum = hashlib.md5(open(zip_path,'rb').read()).hexdigest()
            if dl_checksum != cls._checksum:
                raise ValueError(f"Downloaded deployment package {zip_path} does not match expected checksum")
            with zipfile.ZipFile(zip_path) as zip_ref:
                zip_ref.extractall(os.path.dirname(deploy_dir))
            os.remove(zip_path)
        return deploy_dir


    def __init__(self):
        self.deploy_dir = self.check_pkg()

        self._manifest = None

    def path(self, path):
        """Map paths to be relative to current working directory"""
        return os.path.join(self.deploy_dir, path)

    @property
    def manifest(self):
        if self._manifest is None:
            with open(self.path('manifest.json'), 'r') as f:
                self._manifest = json.load(f)
        return self._manifest

    def __getitem__(self, key):
        return self.manifest[key]

    def __setitem__(self, key, val):
        self.manifest[key] = val


def load_deploy_pkg(for_predict=False, for_filter=False):
    if not (for_predict or for_filter):
        for_predict = True
        for_filter = True

    pkg = DeployPkg()

    ret = list()
    if for_predict:
        tmp_conf_model = dict()
        for lvl_dat in pkg['conf_model']:
            lvl_dat['taxa'] = np.array(lvl_dat['taxa'])

            lvl_dat['model'] = torch.jit.load(pkg.path(lvl_dat.pop('model')))

            tmp_conf_model[lvl_dat['level']] = lvl_dat

        ret.append(torch.jit.load(pkg.path(pkg['nn_model'])))
        ret.append(tmp_conf_model)
        ret.append(pkg['training_config'])
        ret.append("".join(pkg['vocabulary']))

    if for_filter:
        tmp_roc = dict()
        for lvl_dat in pkg['conf_model']:
            tmp_roc[lvl_dat['level']] = np.load(pkg.path(lvl_dat['roc']))
        ret.append(tmp_roc)

    return tuple(ret) if len(ret) > 1 else ret[0]


class GPUModel(nn.Module):

    def __init__(self, model, device):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x).cpu()


def check_cuda(parser):
    if torch.cuda.is_available():
        parser.add_argument('-g', '--gpu', action='store_true', default=False, help='Use GPU')
        parser.add_argument('-D', '--device_id', type=int, default=0,
                            choices=torch.arange(torch.cuda.device_count()).tolist(),
                            help='the device ID of the GPU to use')


def check_device(args):
    if getattr(args, 'gpu', False):
        return torch.device(args.device_id)
    return torch.device('cpu')


def write_csv(output, args):

    # write out data
    if args.output is None:
        outf = sys.stdout
    else:
        outf = open(args.output, 'w')
    output.to_csv(outf, index=True)
