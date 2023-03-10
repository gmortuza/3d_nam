"""
This file will create the mapping scheme for 3dNAM with 8x10 origami
So the origami shape will be 3x8x10. The second layer will be assigned to the parity bits and checksum bits.
The top and bottom layer is for data, indexing and orientation marker. So in total = 160 bits
data + indexing + orientation marker = 160 bits
parity bits + checksum bits = 80 bits
Each parity bits will have XOR of 8 bits, so in total = 640 bits
Each data bit will be present in 4 parity bits so in total = 4 * 160 = 640
"""
import random
from collections import defaultdict

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
PARITY_PERCENT = .3
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
        self.total_parity_bits = int(self.total_bits * parity_percent)
        self.total_check_sum_bits = int(self.total_bits * check_sum_percent)
        self.total_data_bits = self.total_bits - self.total_parity_bits - self.total_check_sum_bits - \
                               self.total_index_bits - self.total_orientation_bits
        if self.total_data_bits <= 0:
            raise Exception("Invalid settings. Total data bits can not be negative. Reduce the parity percent or "
                            "check sum percent.")

        # TODO: calculate the parity coverage and data coverage
        self.parity_coverage, self.data_coverage = self.calculate_parity_and_data_coverage()

    def calculate_parity_and_data_coverage(self):
        return PARITY_COVERAGE, DATA_COVERAGE

    # measure euclidean distance between two points
    @staticmethod
    def get_distance(point1: tuple, point2: tuple) -> float:
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5

    # Measure the distance between a point and a list of points
    def get_distance_multiple_point(self, point: tuple, points: [tuple]) -> float:
        distances = 0
        for p in points:
            distances += self.get_distance(point, p)
        return distances / len(points)

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

    # The bits that are assigned for parity
    def get_parity_bits(self) -> [tuple]:
        # TODO: fix this
        # parity bits is only on the second layer
        parity_bits = []
        for row in range(self.row):
            for column in range(self.column):
                parity_bits.append((LAYER_ASSIGNED_TO_PARITY_BITS, row, column))
        return parity_bits

    # The bits that are assigned for data, indexing and orientation
    def get_data_bits(self):
        # TODO: fix this
        data_bits = []
        layers_assigned_to_data_bits = set(range(self.layer)) - set([LAYER_ASSIGNED_TO_PARITY_BITS])
        for layer in layers_assigned_to_data_bits:
            for row in range(self.row):
                for column in range(self.column):
                    data_bits.append((layer, row, column))
        return data_bits

    # we can not use the same data bits for a parity bit that is mirrored to the other parity bits.
    # This function will return the list of data bits that are not mirrored to the other parity bits
    def get_available_data_bits_for_a_parity_bit(self, parity_bit, data_bits, mapping):
        # A data bit should not be assigned to the same parity bit
        data_bits = set(data_bits)
        mirror_of_parity_bit = self.get_mirror_points_all_axis(parity_bit)
        for mirror_point in mirror_of_parity_bit:
            if mirror_point in mapping:
                data_bits.remove(set(mapping[mirror_point]))
        return data_bits

    # Get all the assigned data bits for a particular parity bit
    def get_assigned_points_for_parity(self, data_bit_counter: dict, available_data_bits: [tuple]) -> [tuple]:
        points = []
        # take a random data bit from the available data bits
        for _ in range(self.parity_coverage):
            if CHOOSE_PARITY_MAPPING_DETERMINISTICALLY:
                available_data_bits_sorted = sorted(list(available_data_bits), key=lambda x: data_bit_counter[x],
                                                    reverse=False)
                data_bit = available_data_bits_sorted[0]
            else:
                available_data_bits_list = list(available_data_bits)
                available_data_bits_probability = [data_bit_counter[i] for i in
                                                   available_data_bits_list]  # picking probability
                if sum(available_data_bits_probability) == 0:
                    available_data_bits_probability = [1 / len(available_data_bits_probability)] * \
                                                      len(available_data_bits_probability)
                data_bit = random.choices(available_data_bits_list, weights=available_data_bits_probability)[0]
            points.append(data_bit)
            # available_data_bits.remove(data_bit)
            # remove the mirror point of this data bits
            mirror_of_data_bit = self.get_mirror_points_all_axis(data_bit)
            # remove the mirror point of this data bits from the available data bits
            available_data_bits = available_data_bits.difference(set(mirror_of_data_bit))
        return points

    # Track number of parity bit assigned to each data bit
    @staticmethod
    def update_counter(data_bit_counter: dict, assigned_data_bits: [tuple]) -> None:
        # fix it
        for data_bit in assigned_data_bits:
            data_bit_counter[data_bit] += 1  # replace inplace

    # Generate the mapping for parity bits
    def get_mapping(self) -> dict:
        # track number of parity that is assigned to each data bit
        data_bit_counter = defaultdict(lambda: 0)
        data_bit_counter = defaultdict(int)
        parity_bit_counter = defaultdict(int)
        check_sum_bit_counter = defaultdict(int)
        parity_bits = self.get_parity_bits()
        data_bits = self.get_data_bits()
        mapping = {}
        # assign random PARITY_COVERAGE data bits to that parity bits
        for parity_bit in parity_bits:  # get data bits for all parity bits
            if parity_bit not in mapping:
                available_data_bits = self.get_available_data_bits_for_a_parity_bit(parity_bit, data_bits, mapping)
                assigned_data_bits = self.get_assigned_points_for_parity(data_bit_counter, available_data_bits)
                mapping[parity_bit] = assigned_data_bits
                self.update_counter(data_bit_counter, assigned_data_bits)
                # Populate the mirror parity bits
                for axis in ['x', 'y', 'xy']:
                    assigned_data_bits_mirrored = self.get_mirror_point(assigned_data_bits, axis)
                    self.update_counter(data_bit_counter, assigned_data_bits_mirrored)
                    mapping[self.get_mirror_point([parity_bit], axis)[0]] = assigned_data_bits_mirrored
        return mapping

    # get distance between parity bit and the assigned data bit
    def get_distance_point_of_mapping(self, mapping):
        distances = {}
        for parity_bit, data_bits in mapping.items():
            distances[parity_bit] = self.get_distance_multiple_point(parity_bit, data_bits)
        return distances

    def get_all_bits(self):
        all_bits = []
        for layer in range(self.layer):
            for row in range(self.row):
                for column in range(self.column):
                    all_bits.append((layer, row, column))
        return all_bits

    # Set the bits purpose(parity, checksum, data, orientation, indexing)
    def determine_bits(self):
        # all bits
        available_bits = self.get_all_bits()
        # first determine the orientation bit
        # for now put four orientation bits
        # first layer four corners
        orientation_bits = [(0, 0, 0), (0, 0, self.column - 1), (0, self.row - 1, 0),
                            (0, self.row - 1, self.column - 1)]
        # remove the orientation bits from the available bits
        available_bits = list(set(available_bits).difference(set(orientation_bits)))
        # get the parity bits
        parity_bits = random.sample(available_bits, self.total_parity_bits)
        # remove the parity bits from the available bits
        available_bits = list(set(available_bits).difference(set(parity_bits)))
        # get the checksum bits
        checksum_bits = random.sample(available_bits, self.total_check_sum_bits)
        # remove the checksum bits from the available bits
        available_bits = list(set(available_bits).difference(set(checksum_bits)))
        # get the indexing bits
        indexing_bits = random.sample(available_bits, self.total_index_bits)
        # indexing, orientation, data will follow a particular order
        indexing_bits.sort()
        # remove the indexing bits from the available bits
        available_bits = list(set(available_bits).difference(set(indexing_bits)))
        # get the data bits
        data_bits = available_bits
        data_bits.sort()
        bits_role = {
            'parity_bits': parity_bits,
            'checksum_bits': checksum_bits,
            'indexing_bits': indexing_bits,
            'data_bits': data_bits,
            'orientation_bits': orientation_bits
        }
        return bits_role

    # Get a point for the mapping based on the distance of the parity bit and the assigned data bit
    def get_mapping_point(self, mapping):
        mapping_point = 0
        distances = self.get_distance_point_of_mapping(mapping)
        for parity_bit, distance in distances.items():
            mapping_point += distance
        return mapping_point / len(mapping.keys())

    def get_optimum_mapping(self):
        highest_point = float('-inf')
        for _ in range(NUMBER_OF_RUN):
            mapping = self.get_mapping()
            mapping_point = self.get_mapping_point(mapping)
            if mapping_point > highest_point:
                highest_point = mapping_point
                best_mapping = mapping
        mapper_details = {
            'parity_mapper': None,
            'checksum_mapper': None,
            'parity_bits': None,
            'checksum_bits': None,
            'data_bits': None,
            'orientation_bit': None,
            'index_bit': None,
        }
        return mapper_details

    def test_get_mirror_point(self):
        points = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        true_mirrored_points_x = [(0, 7, 0), (1, 6, 1), (2, 5, 2)]
        true_mirrored_points_y = [(0, 0, 9), (1, 1, 8), (2, 2, 7)]
        true_mirrored_points_xy = [(0, 7, 9), (1, 6, 8), (2, 5, 7)]
        assert true_mirrored_points_x == self.get_mirror_point(points, 'x')
        assert true_mirrored_points_y == self.get_mirror_point(points, 'y')
        assert true_mirrored_points_xy == self.get_mirror_point(points, 'xy')

    def test(self):
        points = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        self.test_get_mirror_point()


if __name__ == '__main__':
    mapping = Mapping()
    mapping.get_optimum_mapping()
