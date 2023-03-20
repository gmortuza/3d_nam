import math
import multiprocessing
import time
from collections import Counter, defaultdict
from functools import partial

from config import Config
from log import get_logger
from origami_greedy import Origami


class ProcessFile:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(config.verbose, __name__)
        self.origami = Origami(config)
        # get mapping and them to configuration so that it can be used by other classes

    def encode(self, file_in, file_out):
        try:
            file_in = open(file_in, 'rb')
            file_out = open(file_out, "w")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error("Error opening the file")
            return -1, -1, -1, -1  # simulation file expect this format
        data = file_in.read()
        file_in.close()
        # Converting data into binary
        data_in_binary = ''.join(format(letter, '08b') for letter in data)
        # divide the origami based on number of bit per origami

        splitted_data_in_binary = [data_in_binary[i:i + self.config.data_cells_per_origami] for i in
                                   range(0, len(data_in_binary), self.config.data_cells_per_origami)]
        for idx, origami_data in enumerate(splitted_data_in_binary):
            encoded_stream = self.origami.encode(origami_data, idx)
            if self.config.encode_formatted_output:
                print("Matrix -> " + str(idx), file=file_out)
                self.origami.print_matrix(self.origami.data_stream_to_matrix(encoded_stream), in_file=file_out)
            else:
                file_out.write(encoded_stream + '\n')

        file_out.close()
        self.logger.info("Encoding done")

    def single_origami_decode(self, single_origami, ior_file_name, correct_dictionary):
        current_time = time.time()
        self.logger.info("Working on origami(%d): %s", single_origami[0], single_origami[1])
        if len(single_origami[1]) != self.config.layer * self.config.row * self.config.column:
            self.logger.warning("Data point is missing in the origami")
            return
        try:
            decoded_matrix = self.origami.decode(single_origami[1])
        except Exception as e:
            self.logger.exception(e)
            return

        if decoded_matrix == -1:
            return

        self.logger.info("Recovered a origami with index: %s and data: %s", decoded_matrix['index'],
                         decoded_matrix['binary_data'])

        if decoded_matrix['total_probable_error'] > 0:
            self.logger.info("Total %d errors found in locations: %s", decoded_matrix['total_probable_error'],
                             str(decoded_matrix['probable_error_locations']))
        else:
            self.logger.info("No error found")
        # Storing information in individual origami report

        if ior_file_name:
            # Checking correct value
            if correct_dictionary:
                try:
                    status = int(correct_dictionary[int(decoded_matrix['index'])] == decoded_matrix['binary_data'])
                except Exception as e:
                    self.logger.warning(str(e))
                    status = -1
            else:
                status = " "
            decoded_time = round(time.time() - current_time, 3)
            # lock.acquire()
            with open(ior_file_name, "a") as ior_file:
                ior_file.write("{current_origami_index},{origami},{status},{error},{error_location},{orientation},"
                               "{decoded_index},{decoded_origami},{decoded_data},{decoding_time}\n".format(
                                origami=single_origami[1],
                                status=status,
                                error=decoded_matrix['total_probable_error'],
                                error_location=str(decoded_matrix['probable_error_locations']).replace(',', ' '),
                                orientation=decoded_matrix['orientation'],
                                decoded_index=decoded_matrix['index'],
                                decoded_origami=self.origami.matrix_to_data_stream(decoded_matrix['matrix']),
                                decoded_data=decoded_matrix['binary_data'],
                                decoding_time=decoded_time,
                                current_origami_index=single_origami[0]))
            # lock.release()
        return [decoded_matrix, status]

    def decode(self, file_in, file_out):
        correct_origami = 0
        incorrect_origami = 0
        total_error_fixed = 0
        # Read the file
        try:
            data_file = open(file_in, "r")
            data = data_file.readlines()
            data_file.close()
            # File to store individual origami information
            if self.config.write_individual_origami_info:
                ior_file_name = file_out + "_ior.csv"
                with open(ior_file_name, "w") as ior_file:
                    ior_file.write(
                        "Line number in file, origami,status,error,error location,orientation,decoded index,"
                        "decoded origami, decoded data,decoding time\n")
            else:
                ior_file_name = False
        except Exception as e:
            self.logger.error("%s", e)
            return

        # decoded_dictionary = {}
        # If user pass correct file we will create a correct key value pair from that and will compare with our decoded
        # data.
        correct_dictionary = {}
        if self.config.correct_file:
            with open(self.config.correct_file) as cf:
                for so in cf:
                    ci, cd = self.origami.extract_text_and_index(self.data_stream_to_matrix(so.rstrip("\n")))
                    correct_dictionary[ci] = cd
        # Decoded dictionary with number of occurrence of a single origami
        decoded_dictionary_wno = {}
        binary_data_by_origami_index_level = defaultdict(list)
        origami_data = [(i, single_origami.rstrip("\n")) for i, single_origami in enumerate(data)]
        p_single_origami_decode = partial(self.single_origami_decode, ior_file_name=ior_file_name,
                                          correct_dictionary=correct_dictionary)
        if self.config.use_multi_core:
            optimum_number_of_process = int(math.ceil(multiprocessing.cpu_count()))
            pool = multiprocessing.Pool(processes=optimum_number_of_process)
            return_value = pool.map(p_single_origami_decode, origami_data)
            pool.close()
            pool.join()
        else:
            return_value = map(p_single_origami_decode, origami_data)
        for decoded_matrix in return_value:
            if not decoded_matrix is None and not decoded_matrix[0] is None:
                # Checking status
                if self.config.correct_file:
                    if decoded_matrix[1]:
                        correct_origami += 1
                    else:
                        incorrect_origami += 1
                total_error_fixed += int(decoded_matrix[0]['total_probable_error'])
                data_in_binary = decoded_matrix[0]['binary_data']
                for idx_level, binary_val in decoded_matrix[0]['binary_data_by_level'].items():
                    binary_data_by_origami_index_level[idx_level].append(binary_val)

                decoded_dictionary_wno.setdefault(decoded_matrix[0]['index'], []).append(
                    decoded_matrix[0]['binary_data'])

        missing_origami = -1
        # can not check missing origami if file size is not given
        if hasattr(self.config, 'file_size'):
            file_size_in_bit = 8 * self.config.file_size
            origami_needed = math.ceil(file_size_in_bit / file_size_in_bit)
            total_origami_idx_level = set()
            for idx in range(origami_needed):
                for level in self.config.layer:
                    total_origami_idx_level.add(str(idx) + '_' + str(level))
            missing_origami = total_origami_idx_level - set(binary_data_by_origami_index_level.keys())

            if len(missing_origami) > 0:
                return -1, incorrect_origami, correct_origami, total_error_fixed, missing_origami

        # Perform majority voting of binary_data_by_origami_index_level
        final_origami_data = []
        for idx_level in sorted(binary_data_by_origami_index_level.keys()):
            final_origami_data.append(Counter(binary_data_by_origami_index_level[idx_level]).most_common(1)[0][0])

        recovered_binary = "".join(final_origami_data)
        # data was store based on bytes. if recovered data was not multiple of 8 then it was extension part was just
        # for padding, so remove that
        recovered_binary = recovered_binary[:8 * (len(recovered_binary) // 8)]
        # We might put multiple bytes as padding, this won't be removed from previous line of code
        # Remove the padding
        if hasattr(self.config, 'file_size'):
            recovered_binary = recovered_binary[: 8 * self.config.file_size]
        else:
            # keep removing from last 8 bit if all of them are zero
            while len(recovered_binary) >= 8:
                # check last 8 bits
                last_decimal = int(''.join(str(i) for i in recovered_binary[-8:]), 2)
                if last_decimal == 0:  # remove this byte
                    recovered_binary = recovered_binary[:-8]
                else:
                    break

        with open(file_out, "wb") as result_file:
            for start_index in range(0, len(recovered_binary), 8):
                bin_data = recovered_binary[start_index:start_index + 8]
                # convert bin data into decimal
                decimal = int(''.join(str(i) for i in bin_data), 2)
                decimal_byte = bytes([decimal])
                result_file.write(decimal_byte)
        self.logger.info("Number of missing origami :" + str(missing_origami))
        self.logger.info("Total error fixed: " + str(total_error_fixed))
        self.logger.info("File recovery was successful")
        # If we came this far, that means file was recovered correctly
        return 1, incorrect_origami, correct_origami, total_error_fixed, missing_origami


if __name__ == '__main__':
    import filecmp
    FILE_NAME = '../test.txt'
    ENCODED_FILE_NAME = 'encoded.txt'
    DECODED_FILE_NAME = 'decoded.txt'

    config_ = Config('config.yaml')
    process_file = ProcessFile(config_)
    # process_file.encode(FILE_NAME, ENCODED_FILE_NAME)
    process_file.decode(ENCODED_FILE_NAME, DECODED_FILE_NAME)
    # compare the two file
    if filecmp.cmp(FILE_NAME, DECODED_FILE_NAME):
        print("File recovered")
    else:
        print("File was not recovered")

