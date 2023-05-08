import json
import logging
import os
import sys

from pkg_resources import resource_filename


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


class DeployPkg:
    """A class to handle loading and manipulating the deployment package"""

    def __init__(self):
        self.deploy_dir = resource_filename(__name__, 'deploy_pkg')
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
