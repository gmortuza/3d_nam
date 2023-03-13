"""
This file will create the mapping scheme for 3dNAM with 8x10 origami
So the origami shape will be 3x8x10. The second layer will be assigned to the parity bits and checksum bits.
The top and bottom layer is for data, indexing and orientation marker. So in total = 160 bits
data + indexing + orientation marker = 160 bits
parity bits + checksum bits = 80 bits
Each parity bits will have XOR of 8 bits, so in total = 640 bits
Each data bit will be present in 4 parity bits so in total = 4 * 160 = 640
"""
import collections
import random
from collections import defaultdict

import numpy as np

random.seed(0)

# Default settings
# can be updated using function parameters
ROW = 8
COLUMN = 10
LAYER = 3
# Mapping settings
LAYER_ASSIGNED_TO_PARITY_BITS = 1
PARITY_COVERAGE = 8  # Each parity bit will have XOR of 8 data bits
DATA_COVERAGE = 4  # Each data bit will be present in 4 parity bits
NUMBER_OF_RUN = 100  # Number of runs to find the optimum mapping
PARITY_PERCENT = .35
CHECK_SUM_PERCENT = .05
TOTAL_INDEX_BITS = 4
# If True, the data bits will be assigned deterministically, randomly otherwise
CHOOSE_PARITY_MAPPING_DETERMINISTICALLY = False


class Mapping:

    def __init__(self, row=ROW, column=COLUMN, layer=LAYER, parity_percent=PARITY_PERCENT,
                 check_sum_percent=CHECK_SUM_PERCENT, total_index_bits=TOTAL_INDEX_BITS):
        self.row = row
        self.column = column
        self.layer = layer
        self.total_index_bits = total_index_bits

        self.total_orientation_bits = 4
        self.total_bits = row * column * layer

        self.total_parity_cells = self.get_nearest_number_to_match_multiple(self.total_bits * parity_percent)
        self.total_checksum_cells = self.get_nearest_number_to_match_multiple(self.total_bits * check_sum_percent)
        self.total_data_bits = self.total_bits - self.total_parity_cells - self.total_checksum_cells - \
                               self.total_index_bits - self.total_orientation_bits
        self.check_input()
        self.parity_coverage = 8

    def get_nearest_number_to_match_multiple(self, number):
        available_numbers = np.asarray(range(0, self.total_bits, self.layer * self.total_orientation_bits))
        distances = np.abs(available_numbers - number)
        argmin = np.argmin(distances)
        return int(available_numbers[argmin])

    def check_input(self):
        if self.total_data_bits <= 0:
            raise Exception("Invalid settings. Total data bits can not be negative. Reduce the parity percent or "
                            "check sum percent.")
        if self.row % 2 != 0 or self.column % 2 != 0:
            raise Exception("Invalid settings. Row and column must be even number to maintain mirror property.")


    # The given points will be mirrored based on given axis
    def get_mirror_point(self, points: [tuple], axis='x') -> list:
        mirrored_points = []
        for point in points:
            if axis is None:
                mirrored_points.append(point)
            elif axis == 'x':
                mirrored_points.append((point[0], self.row - 1 - point[1], point[2]))
            elif axis == 'y':
                mirrored_points.append((point[0], point[1], self.column - 1 - point[2]))
            elif axis == 'xy':
                mirrored_points.append((point[0], self.row - 1 - point[1], self.column - 1 - point[2]))
        return mirrored_points

    # Given a point, return all the points that are mirrored on all the axis
    def get_mirror_points_all_axis(self, point: tuple):
        mirrored_points = [point]
        for axis in ['x', 'y', 'xy']:
            mirrored_points.extend(self.get_mirror_point([point], axis))
        return mirrored_points


    def get_available_data_cells_for_parity_cell(self, parity_cell, parity_data_map, data_parity_map_by_level):
        available_cells = set(data_parity_map_by_level[parity_cell[0]].keys())
        mirror_of_parity_cell = self.get_mirror_points_all_axis(parity_cell)
        for mirror_point in mirror_of_parity_cell:
            available_cells = available_cells - set(parity_data_map.get(mirror_point, []))
        return list(available_cells)

    def get_random_data_cell_for_parity(self, parity_cell, parity_data_map, data_parity_map_by_level):
        available_cells = self.get_available_data_cells_for_parity_cell(parity_cell, parity_data_map, data_parity_map_by_level)
        sorted_data_cells = sorted(available_cells, key=lambda x: len(data_parity_map_by_level[x[0]][x]))
        return sorted_data_cells[0]

    # Track number of parity bit assigned to each data bit
    @staticmethod
    def update_counter(data_bit_counter: dict, assigned_data_bits: [tuple]) -> None:
        # fix it
        for data_bit in assigned_data_bits:
            data_bit_counter[data_bit] += 1  # replace inplace

    # Generate the mapping for parity bits
    def get_parity_mapping(self, parity_cells, rest_cells) -> dict:
        parity_data_map = {parity_bit: [] for parity_bit in parity_cells}
        data_parity_map_by_level = defaultdict(dict)
        for cell in rest_cells:
            data_parity_map_by_level[cell[0]][cell] = []
        total_parity_map = self.parity_coverage * len(parity_cells) // 4
        for _ in range(total_parity_map):
            # choose a random parity based on least frequency
            # sort the parity based on frequency
            parity_cells_sorted = sorted(parity_cells, key=lambda x: len(parity_data_map[x]), reverse=False)
            parity_cell = parity_cells_sorted[0]
            # get the available data bits for this parity
            random_data_cell = self.get_random_data_cell_for_parity(parity_cell, parity_data_map, data_parity_map_by_level)
            mirrored_parity_cell = self.get_mirror_points_all_axis(parity_cell)
            mirrored_random_data_cell = self.get_mirror_points_all_axis(random_data_cell)
            for parity_cell, random_data_cell in zip(mirrored_parity_cell, mirrored_random_data_cell):
                parity_data_map[parity_cell].append(random_data_cell)
                data_parity_map_by_level[parity_cell[0]][random_data_cell].append(parity_cell)
        return parity_data_map

    def get_checksum_mapping(self, checksum_cells, rest_cells) -> dict:
        # each of the checksum cell will have cell assigned to it
        # cell and checksum will have same levels
        # each cell except for parity cell must be assigned to a checksum cell
        # rest_cells by levels
        rest_cells_by_level = defaultdict(set)
        for cell in rest_cells:
            rest_cells_by_level[cell[0]].add(cell)
        checksum_mapping = defaultdict(list)
        while len(rest_cells_by_level) > 0:
            # don't choose random checksum cell, choose the one with the least number of assigned cells
            checksum_cell = sorted(checksum_cells, key=lambda x: len(checksum_mapping[x]), reverse=False)[0]
            # assign a random cell to the checksum cell
            assigned_cell = random.choice(list(rest_cells_by_level[checksum_cell[0]]))
            checksum_mapping[checksum_cell].append(assigned_cell)
            # remove the assigned cell from the rest cells
            rest_cells_by_level[checksum_cell[0]].remove(assigned_cell)
            # add their mirror points to the checksum mapping
            for axis in ['x', 'y', 'xy']:
                check_sum_mirror_point = self.get_mirror_point([checksum_cell], axis)[0]
                assigned_cell_mirror_point = self.get_mirror_point([assigned_cell], axis)[0]
                checksum_mapping[check_sum_mirror_point].append(assigned_cell_mirror_point)
                # remove the assigned cell from the rest cells
                rest_cells_by_level[checksum_cell[0]].remove(assigned_cell_mirror_point)
            if len(rest_cells_by_level[checksum_cell[0]]) == 0:
                del rest_cells_by_level[checksum_cell[0]]

        return checksum_mapping

    # get distance between parity bit and the assigned data bit
    def get_distance_point_of_mapping(self, mapping):
        distances = {}
        for parity_bit, data_bits in mapping.items():
            distances[parity_bit] = self.get_distance_multiple_point(parity_bit, data_bits)
        return distances

    def get_all_cells(self):
        all_bits = []
        for layer in range(self.layer):
            for row in range(self.row):
                for column in range(self.column):
                    all_bits.append((layer, row, column))
        return all_bits

    def get_random_cells_evenly_distributed_by_levels(self, available_cells, total_cells):
        available_cells_by_layer = defaultdict(set)
        for cell in available_cells:
            available_cells_by_layer[cell[0]].add(cell)
        # each level will have same amount of parity bits
        number_of_cells_per_level = total_cells // self.layer
        # draw a random cell from each level and assign it as a parity bit
        random_cells = []
        while len(random_cells) < total_cells:
            # randomize which layer to choose from
            randomized_layer = list(range(self.layer))
            random.shuffle(randomized_layer)
            for layer in randomized_layer:
                if len(random_cells) >= total_cells:
                    break
                random_cell = random.choice(list(available_cells_by_layer[layer]))
                # get the mirror point of this random cell
                mirror_point = self.get_mirror_points_all_axis(random_cell)
                # remove the mirror point from the available bits
                available_cells_by_layer[layer].difference_update(set(mirror_point))
                random_cells.extend(mirror_point)
        return random_cells

    # Set the bits purpose(parity, checksum, data, orientation, indexing)
    def determine_cells_purpose(self):
        # all bits
        available_cells = self.get_all_cells()
        # available_cells = self.get_all_bits()
        # first determine the orientation bit
        # for now put four orientation bits hardcoded
        # first layer four corners
        orientation_cells = [(0, 0, 0), (0, 0, self.column - 1), (0, self.row - 1, 0),
                             (0, self.row - 1, self.column - 1)]
        # remove the orientation bits from the available bits
        available_cells = list(set(available_cells).difference(set(orientation_cells)))
        # get the parity bits
        parity_cells = self.get_random_cells_evenly_distributed_by_levels(available_cells, self.total_parity_cells)
        # remove the parity bits from the available bits
        available_cells = list(set(available_cells).difference(set(parity_cells)))
        # get the checksum bits
        checksum_cells = self.get_random_cells_evenly_distributed_by_levels(available_cells, self.total_checksum_cells)
        # remove the checksum bits from the available bits
        available_cells = list(set(available_cells).difference(set(checksum_cells)))
        # get the indexing bits
        indexing_cells = self.get_random_cells_evenly_distributed_by_levels(available_cells, self.total_index_bits)
        # indexing, orientation, data will follow a particular order
        indexing_cells.sort()
        # remove the indexing bits from the available bits
        available_cells = list(set(available_cells).difference(set(indexing_cells)))
        # get the data bits
        data_cells = available_cells
        data_cells.sort()
        return {
            'parity_cells': parity_cells,
            'checksum_cells': checksum_cells,
            'indexing_cells': indexing_cells,
            'data_cells': data_cells,
            'orientation_cells': orientation_cells,
            'all_but_parity': checksum_cells + indexing_cells + data_cells + orientation_cells,
            'all_but_parity_checksum': indexing_cells + data_cells + orientation_cells,
        }

    @staticmethod
    def get_2d_std(points):
        points = np.array(points)
        x = points[:, 1]
        y = points[:, 2]
        x_std = np.abs(x - np.mean(x))
        y_std = np.abs(y - np.mean(y))
        combined = x_std + y_std
        return sum(combined)

    # Get a point for the mapping based on the distance of the parity bit and the assigned data bit
    def get_mapping_point(self, mapping):
        total_std = 0
        for parity_bit, data_bits in mapping.items():
            total_std += self.get_2d_std(data_bits)
        return total_std / len(mapping.keys())

    def get_optimum_mapping(self):
        # define the purpose of each bit(parity, checksum, data, orientation, indexing)
        highest_point = float('-inf')
        for _ in range(NUMBER_OF_RUN):
            cell_purpose = self.determine_cells_purpose()
            # get parity mapper
            current_parity_mapping = self.get_parity_mapping(cell_purpose['parity_cells'],
                                                             cell_purpose['all_but_parity'])
            # get checksum mapping
            current_checksum_mapping = self.get_checksum_mapping(cell_purpose['checksum_cells'],
                                                                 cell_purpose['all_but_parity_checksum'])

            current_parity_mapping_point = self.get_mapping_point(current_parity_mapping)
            current_checksum_mapping_point = self.get_mapping_point(current_checksum_mapping)
            total_mapping_point = current_parity_mapping_point + current_checksum_mapping_point

            if total_mapping_point > highest_point:
                highest_point = total_mapping_point
                best_checksum_mapping = current_checksum_mapping
                best_parity_mapping = current_parity_mapping
                best_cell_purpose = cell_purpose
        # combine the best mapping and the best bit purpose
        return {
            'parity_mapping': best_parity_mapping,
            'checksum_mapping': best_checksum_mapping,
            'cell_purpose': best_cell_purpose,
        }


if __name__ == '__main__':
    mapping_ = Mapping()
    optimum_mapping = mapping_.get_optimum_mapping()
