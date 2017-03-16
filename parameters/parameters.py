import json
import os


class Parameters:
    params = {}

    def __init__(self):
        self.params_file = "../configs/parameters.json"

    def write_to_parameters_file(self, new_params_file):
        json.dump(self.params, open(new_params_file, 'w'), indent=4)

    def read_from_params_file(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(current_path, self.params_file)
        if os.path.exists(file_name):
            self.params = json.load(open(file_name))
        else:
            raise ValueError("The path \"" + os.path.join(current_path, self.params_file) +
                             "\" does not exist. current path: " + current_path)
