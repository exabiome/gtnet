import hashlib
import json
import logging
import os
import sys
import urllib
import warnings
import zipfile

from pkg_resources import resource_filename


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
        if not os.path.exists(deploy_dir):
            msg = ("Downloading GTNet deployment package. This will only happen on the first invocation "
                   "of gtnet predict")
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
