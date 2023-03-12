import collections

from error_correction.find_optimum_mapping import Mapping


def test_get_mirror_point(mapping_obj):
    points = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    true_mirrored_points_x = [(0, 7, 0), (1, 6, 1), (2, 5, 2)]
    true_mirrored_points_y = [(0, 0, 9), (1, 1, 8), (2, 2, 7)]
    true_mirrored_points_xy = [(0, 7, 9), (1, 6, 8), (2, 5, 7)]
    assert true_mirrored_points_x == mapping_obj.get_mirror_point(points, 'x')
    assert true_mirrored_points_y == mapping_obj.get_mirror_point(points, 'y')
    assert true_mirrored_points_xy == mapping_obj.get_mirror_point(points, 'xy')


def test_parity_mapping(mapping_object, parity_mapping):
    total_data = []
    for parity_cell, parity_coverage in parity_mapping.items():
        total_data.extend(parity_coverage)
        assert len(parity_coverage) == mapping_object.parity_coverage
    counter = collections.Counter(total_data)

    pass


def test_checksum_mapping(mapping_object, checksum_mapping):
    pass


def test_cell_purpose_length(mapping_object, cell_distribution):
    # we must have 4 orientations cell
    assert len(cell_distribution['orientation_cells']) == 4
    # divisor layer * 4. 4 is for maintaining the mirror property
    divisor = mapping_object.layer * 4
    assert len(cell_distribution['parity_cells']) % divisor == 0
    assert len(cell_distribution['checksum_cells']) % divisor == 0
    assert len(cell_distribution['data_cells'] + cell_distribution['indexing_cells'] +
               cell_distribution['orientation_cells']) % divisor == 0

    # check total number of cells
    total_cells = mapping_object.row * mapping_object.column * mapping_object.layer
    assert len(cell_distribution['parity_cells']) + len(cell_distribution['checksum_cells']) + len(
        cell_distribution['data_cells'] + cell_distribution['orientation_cells'] +
        cell_distribution['indexing_cells']) == total_cells

    assert len(cell_distribution['parity_cells']) + len(cell_distribution['all_but_parity']) == total_cells
    assert len(cell_distribution['parity_cells']) + len(cell_distribution['checksum_cells']) + \
           len(cell_distribution['all_but_parity_checksum']) == total_cells


def test_cell_purpose_mirror_property(mapping_object, cell_distribution):
    # parity mirror property
    all_parity_cells = set(cell_distribution['parity_cells'])
    for parity_cell in cell_distribution['parity_cells']:
        for axis in ['x', 'y', 'xy']:
            mirror_point = mapping_object.get_mirror_point([parity_cell], axis)[0]
            assert mirror_point in all_parity_cells

    # checksum mirror property
    all_checksum_cells = set(cell_distribution['checksum_cells'])
    for checksum_cell in cell_distribution['checksum_cells']:
        for axis in ['x', 'y', 'xy']:
            mirror_point = mapping_object.get_mirror_point([checksum_cell], axis)[0]
            assert mirror_point in all_checksum_cells


def test_cell_purpose(mapping_object, cell_distribution):
    test_cell_purpose_length(mapping_object, cell_distribution)
    test_cell_purpose_mirror_property(mapping_object, cell_distribution)


def test_mapping_mirror_property(mapping_object, mapping):
    # convert mapping to set
    pass

def test_mapping_details(mapping_object, mapping_details):
    test_cell_purpose(mapping_object, mapping_details['cell_purpose'])
    test_mapping_mirror_property(mapping_object, mapping_details['parity_mapping'])
    test_mapping_mirror_property(mapping_object, mapping_details['checksum_mapping'])
    test_parity_mapping(mapping_object, mapping_details['parity_mapping'])
    test_checksum_mapping(mapping_object, mapping_details['checksum_mapping'])


def main():
    mapping_object = Mapping()
    mapping_details = mapping_object.get_optimum_mapping()

    test_get_mirror_point(mapping_object)
    test_mapping_details(mapping_object, mapping_details)


if __name__ == '__main__':
    main()
