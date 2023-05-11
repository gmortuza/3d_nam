import numpy as np


def print_matrix(config, matrix, in_file=False):
    """
    Display a given matrix

    :param: matrix: A 3-D matrix
    :param: in_file: if we want to save the encoding information in a file.

    :returns: None
    """
    for layer in range(config.layer):
        if not in_file:
            print("Layer: ", layer, end="\n")
        else:
            print("Layer: ", layer, end="\n", file=in_file)
        for row in range(config.row):
            for column in range(config.column):
                if not in_file:
                    print(matrix[layer][row][column], end="\t")
                else:
                    print(matrix[layer][row][column], end="\t", file=in_file)
            if not in_file:
                print("")
            else:
                print("", file=in_file)


def matrix_to_data_stream(config, matrix):
    """
    Convert 2-D matrix to string

    :param: matrix: A 2-D matrix
    :returns: data_stream: string of 2-D matrix
    """
    data_stream = []
    for level in range(config.layer):
        for row in range(config.row):
            for column in range(config.column):
                data_stream.append(matrix[level][row][column])
    return ''.join(str(i) for i in data_stream)


def data_stream_to_matrix(config, data_stream):
    """
    Convert a sting to 3-D matrix

    The length of data stream should be 48 bit currently this algorithm is only working with 6x8 matrix

    :param: data_stream: 48 bit of string
    :returns: matrix: return 3-D matrix
    """
    return np.asarray(list(map(int, list(data_stream)))).reshape(
        (config.layer, config.row, config.column))
