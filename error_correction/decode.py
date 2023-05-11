import argparse

from error_correction.config import Config
from error_correction.process_file_new import ProcessFile


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
    args = parser.parse_args()
    return args


def main():
    args = read_args()
    config = Config(args.config)
    dnam_decode = ProcessFile(config)
    dnam_decode.decode(args.file_in, args.file_out)


if __name__ == '__main__':
    main()
