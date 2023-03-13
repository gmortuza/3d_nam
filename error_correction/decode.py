import argparse

from error_correction.config import Config
from processfile import ProcessFile


def read_args():
    """
    Read the arguments from command line
    :return:
    """
    parser = argparse.ArgumentParser(description="Decode a given origami matrices to a text file.")
    parser.add_argument("-f", "--file_in", help="File to decode", required=True)
    parser.add_argument("-o", "--file_out", help="File to write output", required=True)
    parser.add_argument("-fz", "--file_size", help="File size that will be decoded", type=int, required=True)
    parser.add_argument("-c", "--config", help="Config file", type=str, default='config.yaml')
    # parser.add_argument('-tp', '--threshold_parity',
    #                     help='Minimum weight for a parity bit cell to be consider that as an error', default=2, type=int)
    # parser.add_argument("-td", "--threshold_data",
    #                     help='Minimum weight for a data bit cell to be consider as an error', default=2, type=int)
    # parser.add_argument("-v", "--verbose", help="Print details on the console. "
    #                                             "0 -> error, 1 -> debug, 2 -> info, 3 -> warning", default=0, type=int)
    # parser.add_argument("-r", "--redundancy", help="How much redundancy was used during encoding",
    #                     default=50, type=float)
    # parser.add_argument("-ior", "--individual_origami_info", help="Store individual origami information",
    #                     action='store_true', default=True)
    # parser.add_argument("-e", "--error", help="Maximum number of error that the algorithm "
    #                                           "will try to fix", type=int, default=8)
    # parser.add_argument("-fp", "--false_positive", help="0 can also be 1.", type=int, default=0)
    #
    # parser.add_argument("-d", "--degree", help="Degree old/new", default="new", type=str)
    #
    # parser.add_argument("-cf", "--correct_file", help="Original encoded file. Helps to check the status automatically."
    #                     , type=str, default=False)
    # parser.add_argument('-p', '--parallelism', help='Use multiple process', action='store_true', default=False)
    #
    args = parser.parse_args()
    return args


def main():
    args = read_args()
    config = Config(args.config)
    dnam_decode = ProcessFile(config)
    dnam_decode.decode(args.file_in, args.file_out, args.file_size)


if __name__ == '__main__':
    main()
