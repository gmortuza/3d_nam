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

# Origami settings
ROW = 8
COLUMN = 10
LAYER = 3
# Mapping settings
LAYER_ASSIGNED_TO_PARITY_BITS = 1
PARITY_COVERAGE = 8  # Each parity bit will have XOR of 8 data bits
DATA_COVERAGE = 4  # Each data bit will be present in 4 parity bits
NUMBER_OF_RUN = 100  # Number of runs to find the optimum mapping
CHOOSE_PARITY_MAPPING_DETERMINISTICALLY = False  # If True, the data bits will be assigned deterministically, randomly otherwise


# measure euclidean distance between two points
def get_distance(point1: tuple, point2: tuple) -> float:
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5


# Measure the distance between a point and a list of points
def get_distance_multiple_point(point: tuple, points: [tuple]) -> float:
    distances = 0
    for p in points:
        distances += get_distance(point, p)
    return distances / len(points)


# The given points will be mirrored based on given axis
def get_mirror_point(points: [tuple], axis='x') -> list:
    mirrored_points = []
    for point in points:
        if axis is None:
            mirrored_points.append(point)
        elif axis == 'x':
            mirrored_points.append((point[0], ROW - 1 - point[1], point[2]))
        elif axis == 'y':
            mirrored_points.append((point[0], point[1], COLUMN - 1 - point[2]))
        elif axis == 'xy':
            mirrored_points.append((point[0], ROW - 1 - point[1], COLUMN - 1 - point[2]))
    return mirrored_points


# Given a point, return all the points that are mirrored on all the axis
def get_mirror_points_all_axis(point: tuple):
    mirrored_points = [point]
    for axis in ['x', 'y', 'xy']:
        mirrored_points.extend(get_mirror_point([point], axis))
    return mirrored_points


# The bits that are assigned for parity
def get_parity_bits() -> [tuple]:
    # parity bits is only on the second layer
    parity_bits = []
    for row in range(ROW):
        for column in range(COLUMN):
            parity_bits.append((LAYER_ASSIGNED_TO_PARITY_BITS, row, column))
    return parity_bits


# The bits that are assigned for data, indexing and orientation
def get_data_bits():
    data_bits = []
    layers_assigned_to_data_bits = set(range(LAYER)) - set([LAYER_ASSIGNED_TO_PARITY_BITS])
    for layer in layers_assigned_to_data_bits:
        for row in range(ROW):
            for column in range(COLUMN):
                data_bits.append((layer, row, column))
    return data_bits

# we can not use the same data bits for a parity bit that is mirrored to the other parity bits.
# This function will return the list of data bits that are not mirrored to the other parity bits
def get_available_data_bits_for_a_parity_bit(parity_bit, data_bits, mapping):
    # A data bit should not be assigned to the same parity bit
    data_bits = set(data_bits)
    mirror_of_parity_bit = get_mirror_points_all_axis(parity_bit)
    for mirror_point in mirror_of_parity_bit:
        if mirror_point in mapping:
            data_bits.remove(set(mapping[mirror_point]))
    return data_bits

# Get all the assigned data bits for a particular parity bit
def get_assigned_points_for_parity(data_bit_counter: dict, available_data_bits: [tuple]) -> [tuple]:
    points = []
    # take a random data bit from the available data bits
    for _ in range(PARITY_COVERAGE):
        if CHOOSE_PARITY_MAPPING_DETERMINISTICALLY:
            available_data_bits_sorted = sorted(list(available_data_bits), key=lambda x: data_bit_counter[x],
                                                reverse=False)
            data_bit = available_data_bits_sorted[0]
        else:
            available_data_bits_list = list(available_data_bits)
            available_data_bits_probability = [data_bit_counter[i] for i in available_data_bits_list]  # picking probability
            data_bit = random.choices(available_data_bits_list, weights=available_data_bits_probability)[0]
        points.append(data_bit)
        # available_data_bits.remove(data_bit)
        # remove the mirror point of this data bits
        mirror_of_data_bit = get_mirror_points_all_axis(data_bit)
        # remove the mirror point of this data bits from the available data bits
        available_data_bits = available_data_bits.difference(set(mirror_of_data_bit))
    return points

# Track number of parity bit assigned to each data bit
def update_counter(data_bit_counter: dict, assigned_data_bits: [tuple]) -> None:
    for data_bit in assigned_data_bits:
        data_bit_counter[data_bit] += 1  # repalce inplace


# Generate the mapping for parity bits
def get_mapping() -> dict:
    # track number of parity that is assigned to each data bit
    data_bit_counter = defaultdict(lambda: 0)
    mapping = {}
    # get the parity bits
    parity_bits = get_parity_bits()
    # get the data bits
    data_bits = get_data_bits()
    # assign random PARITY_COVERAGE data bits to that parity bits
    for parity_bit in parity_bits:  # get data bits for all parity bits
        if parity_bit not in mapping:
            available_data_bits = get_available_data_bits_for_a_parity_bit(parity_bit, data_bits, mapping)
            assigned_data_bits = get_assigned_points_for_parity(data_bit_counter, available_data_bits)
            mapping[parity_bit] = assigned_data_bits
            update_counter(data_bit_counter, assigned_data_bits)
            # Populate the mirror parity bits
            for axis in ['x', 'y', 'xy']:
                assigned_data_bits_mirrored = get_mirror_point(assigned_data_bits, axis)
                update_counter(data_bit_counter, assigned_data_bits_mirrored)
                mapping[get_mirror_point([parity_bit], axis)[0]] = assigned_data_bits_mirrored
    return mapping


# get distance between parity bit and the assigned data bit
def get_distance_point_of_mapping(mapping):
    distances = {}
    for parity_bit, data_bits in mapping.items():
        distances[parity_bit] = get_distance_multiple_point(parity_bit, data_bits)
    return distances

# Get a point fro the mapping based on the distance of the parity bit and the assigned data bit
def get_mapping_point(mapping):
    mapping_point = 0
    distances = get_distance_point_of_mapping(mapping)
    for parity_bit, distance in distances.items():
        mapping_point += distance
    return mapping_point / len(mapping.keys())


def main():
    for _ in range(NUMBER_OF_RUN):
        mapping = get_mapping()
        mapping_point = get_mapping_point(mapping)
        print(mapping_point)


def test_get_mirror_point():
    points = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    true_mirrored_points_x = [(0, 7, 0), (1, 6, 1), (2, 5, 2)]
    true_mirrored_points_y = [(0, 0, 9), (1, 1, 8), (2, 2, 7)]
    true_mirrored_points_xy = [(0, 7, 9), (1, 6, 8), (2, 5, 7)]
    assert true_mirrored_points_x == get_mirror_point(points, 'x')
    assert true_mirrored_points_y == get_mirror_point(points, 'y')
    assert true_mirrored_points_xy == get_mirror_point(points, 'xy')


def test():
    points = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    test_get_mirror_point()


if __name__ == '__main__':
    main()
