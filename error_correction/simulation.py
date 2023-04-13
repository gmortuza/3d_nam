import sys
import numpy as np
import time
import os
import filecmp
import datetime
import random
import argparse
from error_correction.config import Config
import utils

sys.path.append("../dnam/")

# TODO: Revert to process file
from error_correction.process_file_new import ProcessFile
from error_correction.log import get_logger

parser = argparse.ArgumentParser(description="Simulation file for 3dNAM")
parser.add_argument("-s", "--start", help="Starting file size in bytes", type=int, default=20)
parser.add_argument("-e", "--end", help="Ending file size in bytes", type=int, default=2000)
parser.add_argument("-g", "--gap", help="Gap between file size", type=int, default=40)
parser.add_argument("-ec", "--error_check", help="Maxiumum number of error that will be checked for each origami",
                    type=int, default=12)
parser.add_argument("-oc", "--copies_of_each_origami", help="A single origami will have multiple copy with different"
                                                            " level of error", type=int, default=10)
parser.add_argument("-fp", "--false_positive_per_origami", help="Number of false positive error that will be "
                                                                "added in each origami", type=int, default=0)
parser.add_argument("-a", "--average", help="The error will be distributed on average per origami", action="store_true")
parser.add_argument("-v", "--verbose", help="Print all the details", type=int, default=0)
args = parser.parse_args()

STARTING_FILE_SIZE = args.start  # Starting file size in bytes for simulation
ENDING_FILE_SIZE = args.end  # Ending file size in bytes for simulation
FILE_SIZE_GAP = args.gap  # Interval size of the file for simulation
MAXIMUM_NUMBER_OF_ERROR_CHECKED_PER_ORIGAMI = args.error_check
MAXIMUM_NUMBER_OF_ERROR_INSERTED_PER_ORIGAMI = args.error_check
COPIES_OF_EACH_ORIGAMIES = args.copies_of_each_origami
FALSE_POSITIVE_PER_ORIGAMI = args.false_positive_per_origami
AVERAGE = args.average
VERBOSE = args.verbose
ROW_NUMBER = 8
COLUMN_NUMBER = 10

RESULT_DIRECTORY = "test_result"
DATA_DIRECTORY = "test_data"
CURRENT_DATE = str(datetime.datetime.now()).replace("-", "")[:8]
SIMULATION_DIRECTORY = RESULT_DIRECTORY + "/" + CURRENT_DATE + "_simulation_" + str(STARTING_FILE_SIZE) + "_" + str(
    ENDING_FILE_SIZE)
RESULT_FILE_NAME = SIMULATION_DIRECTORY + "/overall_result_simulation.csv"

logger = get_logger(VERBOSE, __name__)


def set_random_configuration(config):
    # get random row column level
    config.row = random.choice(config.sim_rows)
    config.column = random.choice(config.sim_cols)
    config.layer = random.choice(config.sim_layers)
    config.parity_coverage = random.choice(config.sim_parity_coverage)
    config.parity_percent = random.choice(config.sim_parity_percent)
    config.checksum_percent = random.choice(config.sim_checksum_percent)
    config.create_mapping()
    config.calculate_additional_parameters()


def change_orientation(config, origamies):
    """
    This will randomly alter the origami orientation
    :param origamies:
    :return:
    """
    for i, single_origami in enumerate(origamies):
        orientation_option = random.choice(range(4))
        # orientation_option = 0
        single_matrix = utils.data_stream_to_matrix(config, single_origami.rstrip('\n'))
        if orientation_option == 0:
            single_matrix = single_matrix
        elif orientation_option in (1, 2):
            single_matrix = np.flip(single_matrix, axis=orientation_option)
        else:
            single_matrix = np.flip(np.flip(single_matrix, axis=2), axis=1)
        origamies[i] = single_matrix
    logger.info("Origami has been randomly oriented")
    return origamies


def degrade(file_in, file_out, number_of_error, config):
    try:
        degrade_file = open(file_out, "w")
        encoded_file = open(file_in, "r")
    except Exception as e:
        logger.error(e)
    total_error_inserted = 0
    origami_number = 0
    encode_file_list = encoded_file.readlines() * config.sim_copies_of_each_origamies
    random.shuffle(encode_file_list)
    encode_file_list = change_orientation(config, encode_file_list)
    total_origami = len(encode_file_list)
    total_error = int(number_of_error * total_origami)
    total_false_positive_errors = int(total_error * config.sim_fp_error_percent)  # 0 --> 1
    total_true_positive_errors = total_error - total_false_positive_errors  # 1 --> 0

    false_positive_inserted = 0
    # get all cells that are 1
    encode_file_arr = np.asarray(encode_file_list)
    # Error will be introduced on average per origami
    if not config.sim_average_error_dist:
        # get random index to insert error
        one_cells = np.argwhere(encode_file_arr == 1)
        random_index = random.sample(range(len(one_cells)), total_true_positive_errors)
        for idx in random_index:
            i, l, r, c = one_cells[idx]
            encode_file_list[i][l][r][c] = 0
            total_error_inserted += 1
    # Each origami will be fixed amount of error
    else:
        # evenly distribute errors
        number_of_error = int(total_true_positive_errors / (total_origami * config.layer))
        for index in range(len(encode_file_list)):
            for layer in range(config.layer):
                one_cells = np.argwhere(encode_file_arr[index][layer] == 1)
                random_index = random.sample(range(len(one_cells)), number_of_error)
                for idx in random_index:
                    r, c = one_cells[idx]
                    encode_file_list[index][layer][r][c] = 0
                    total_error_inserted += 1

    zero_cells = np.argwhere(encode_file_arr == 0)
    random_index = random.sample(range(len(zero_cells)), total_false_positive_errors)
    for idx in random_index:
        i, l, r, c = zero_cells[idx]
        encode_file_list[i][l][r][c] = 1
        total_error_inserted += 1

    for single_origami in encode_file_list:
        degraded_origami = utils.matrix_to_data_stream(config, single_origami)
        degrade_file.write(degraded_origami + "\n")
        origami_number += 1
    degrade_file.close()
    encoded_file.close()
    logger.info("Error insertion done")
    return total_error_inserted


def create_result_file(config):
    # Create the simulation directory if it's not there already
    if not os.path.isdir(SIMULATION_DIRECTORY):
        os.makedirs(SIMULATION_DIRECTORY)
    # Result file name
    # create result file
    try:
        # If file exists we will just append the result otherwise create new file
        if os.path.isfile(RESULT_FILE_NAME):
            result_file = open(RESULT_FILE_NAME, "a")
        else:
            result_file = open(RESULT_FILE_NAME, "w")
            result_file.write("File size,row,column,layer,parity percent,checksum percent,parity coverage,"
                              "Number of copies of each origami,Maximum number of error checked per origami,"
                              "Encoding time,Decoding time,Number of error per origami,Total number of error,"
                              "Total number of error detected,Incorrect origami,Correct Origami,Missing origamies,status,threshold data,"
                              "threshold parity,false positive\n")
        # Closing the file otherwise it will be copied on the multi processing
        # Each process copied all the open file descriptor. If this we don't close this file here.
        # Then this file descriptor  will be copied by each process hence memory consumption will
        # be more / we might get the error of too many file open
        # So we will close this here before we call the multiprocessing
        result_file.close()
    except Exception as e:
        logger.error("Couldn't open the result file")
        logger.exception(e)
        exit()


def run_simulation(config):
    for file_size in list(range(config.sim_file_size[0], config.sim_file_size[1], config.sim_file_size[2])):
        config.file_size = file_size
        test_file_name = SIMULATION_DIRECTORY + "/test_" + str(file_size)
        logger.info("working with file size: {file_size}".format(file_size=file_size))
        # Generate random binary file for encoding
        with open(test_file_name, "wb", 0) as random_file:
            random_file.write(os.urandom(file_size))
        # encode the randomly generated file
        dnam_object = ProcessFile(config)
        encoded_file_name = test_file_name + "_encode"
        start_time = time.time()
        # Encode the file
        dnam_object.encode(test_file_name, encoded_file_name)
        config.correct_file = encoded_file_name
        encoding_time = round((time.time() - start_time), 2)
        for error_in_each_origami in range(4, config.maximum_error_to_fix + 1):
            error_in_each_origami = round(error_in_each_origami * config.layer, 2)
            logger.info(f"Checking error: {error_in_each_origami}")
            degraded_file_name = encoded_file_name + "_degraded_copy_" + str(
                config.sim_copies_of_each_origamies) + "_error_" + str(error_in_each_origami)
            # if error_in_each_origami == 0:
            total_error_insertion = degrade(encoded_file_name, degraded_file_name, error_in_each_origami, config)
            logger.info("Degradation done")
            dnam_decode = ProcessFile(config)
            # try to decode with different decoding parameter
            for threshold_data in range(2, 4):  # This two loops are for the parameter.
                for threshold_parity in range(2, 4):  # Now we are choosing only one parameter.
                    config.data_threshold = threshold_data
                    config.parity_threshold = threshold_parity
                    decoded_file_name = test_file_name + "_decoded_copy_" + str(config.sim_copies_of_each_origamies) + "_error_" + \
                                        str(error_in_each_origami) + "_scp_" + str(threshold_data) + \
                                        "_tempweight_" + str(threshold_parity)
                    start_time = time.time()
                    try:
                        decoding_status, incorrect_origami, correct_origami, total_error_fixed, missing_origamies \
                            = dnam_decode.decode(degraded_file_name, decoded_file_name)
                        if os.path.exists(encoded_file_name) and os.path.exists(decoded_file_name) and filecmp.cmp(
                                test_file_name, decoded_file_name):
                            status = 1
                        else:
                            if decoding_status == -1:
                                status = -1  # We could detect
                            else:
                                status = 0  # We couldn't detect
                                print("Couldn't detect")
                    except Exception as e:
                        print(e)  # Something went wrong on the decoding
                        status = -2
                        incorrect_origami = -1
                        correct_origami = -1
                        total_error_fixed = -1
                        missing_origamies = []
                    decoding_time = round((time.time() - start_time), 2)
                    with open(RESULT_FILE_NAME, "a") as result_file:
                        result_file.write(f"{file_size},{config.row},{config.column},{config.layer},{config.parity_percent},"
                                          f"{config.checksum_percent},{config.parity_coverage},"
                                          f"{config.sim_copies_of_each_origamies},"
                                          f"{config.sim_maximum_number_of_error_checked_per_origami},{encoding_time},{decoding_time},{error_in_each_origami},{total_error_insertion},"
                                          f"{total_error_fixed},{incorrect_origami},{correct_origami},{str(list(missing_origamies)).replace(',', ' ')},{status},{config.data_threshold},{config.parity_threshold},{config.sim_fp_error_percent}\n")
                    if os.path.exists(decoded_file_name) and status == 1:
                        os.remove(decoded_file_name)
                        pass
            del dnam_decode
            if status == 1:  # if we can decode the file we will remove that. Other wise keep it for future debug reference.
                os.remove(degraded_file_name)
        del dnam_object  # clearing up the memory


def main(config):
    # update the config file
    set_random_configuration(config)
    create_result_file(config)
    run_simulation(config)


if __name__ == '__main__':
    config_ = Config('../error_correction/config.yaml')
    main(config_)
