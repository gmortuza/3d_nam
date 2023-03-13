import argparse

from error_correction.config import Config
from processfile import ProcessFile
import os


def read_args():
    """
    Read argument from command line
    :return: arguments
    """
    parser = argparse.ArgumentParser(description="Encode a given file to a list of origami matrices.")
    parser.add_argument("-f", "--file_in", help="file to encode", required=True)
    parser.add_argument("-o", "--file_out", help="File to write the output", required=True)
    parser.add_argument('-c', '--config', help='Config file', default='config.yaml', type=str)
    # parser.add_argument("-r", "--redundancy", help="Percentage of redundant origami", default=50, type=float)
    # parser.add_argument("-fo", "--formatted_output", help="Will print the origami as matrix instead of single line",
    #                     action="store_true")
    # parser.add_argument("-v", "--verbose", help="Print details on the console. "
    #                                             "0 -> error. 1->debug, 2->info, 3->warning", default=0, type=int)
    # parser.add_argument("-d", "--degree", help="Degree old/new", default="new", type=str)
    args = parser.parse_args()
    return args


def main():
    """
    The main program that will run
    :return: None
    """
    args = read_args()
    config = Config(args.config)
    dnam_object = ProcessFile(config)
    dnam_object.encode(os.path.abspath(args.file_in), os.path.abspath(args.file_out))


if __name__ == '__main__':
    main()
