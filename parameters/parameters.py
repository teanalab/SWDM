import json
import os

import sys


class Parameters:
    def __init__(self):
        self.params_file = "../configs/parameters.json"
        self.params = {}

    def write_to_parameters_file(self, new_params_file):
        json.dump(self.params, open(new_params_file, 'w'), indent=4)

    def read_from_params_file(self):
        if os.path.exists(self.params_file):
            self.params = json.load(open(self.params_file))
