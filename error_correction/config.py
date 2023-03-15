from collections import defaultdict, Counter

import yaml

from error_correction.find_optimum_mapping import Mapping


class Config:
    def __init__(self, config_file_path=None):
        # if config path is none search for config.yaml in the current directory
        self.data_to_parity_mapping_by_level = None
        self.data_cells_per_origami = None
        self.data_to_parity_mapping = None
        self.data_cells_per_level = None
        config_file_path = 'config.yaml' if config_file_path is None else config_file_path
        self.parity_mapping = None
        self.checksum_mapping = None
        self.cell_purpose = None
        self.checksum_mapping_by_level = None
        self.parity_mapping_by_level = None
        self.total_cell = None
        self.orientation_data = [1, 1, 1, 0]
        self.update(config_file_path)
        # create the mapping as it will require from both encode and decode
        self.create_mapping()
        self.calculate_additional_parameters()

    def update(self, config_file_path):
        with open(config_file_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

    def create_mapping(self):
        mapping_details = Mapping(self).get_optimum_mapping()
        # add the mapping to the config
        self.parity_mapping = mapping_details['parity_mapping']
        self.checksum_mapping = mapping_details['checksum_mapping']
        self.cell_purpose = mapping_details['cell_purpose']

    def calculate_additional_parameters(self):
        self.total_cell = self.layer * self.row * self.column
        # data bits to parity bits mapping
        self.data_to_parity_mapping = defaultdict(list)
        for parity_bit, data_bits in self.parity_mapping.items():
            for data_bit in data_bits:
                self.data_to_parity_mapping[data_bit].append(parity_bit)

        # data bits per level
        self.data_cells_per_level = Counter()
        for data_cell in self.cell_purpose['data_cells']:
            self.data_cells_per_level[data_cell[0]] += 1

        # data cells per origami
        self.data_cells_per_origami = len(self.cell_purpose['data_cells'])

        # Level wise mapping
        self.parity_mapping_by_level = defaultdict(lambda: defaultdict(list))
        self.checksum_mapping_by_level = defaultdict(lambda: defaultdict(list))
        self.data_to_parity_mapping_by_level = defaultdict(lambda: defaultdict(list))

        for parity_bit, data_bits in self.parity_mapping.items():
            # data bits without level
            data_bits_without_level = [data_bit[1:] for data_bit in data_bits]
            self.parity_mapping_by_level[parity_bit[0]][(parity_bit[1], parity_bit[2])] = data_bits_without_level

        for checksum_bit, data_bits in self.checksum_mapping.items():
            data_bits_without_level = [data_bit[1:] for data_bit in data_bits]
            self.checksum_mapping_by_level[checksum_bit[0]][(checksum_bit[1], checksum_bit[2])] = data_bits_without_level

        for data_bit, parity_bits in self.data_to_parity_mapping.items():
            parity_bits_without_level = [parity_bit[1:] for parity_bit in parity_bits]
            self.data_to_parity_mapping_by_level[data_bit[0]][(data_bit[1], data_bit[2])] = parity_bits_without_level


if __name__ == '__main__':
    config = Config()
    print(config.dict)
