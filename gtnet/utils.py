import os
from pkg_resources import resource_filename
import json


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
