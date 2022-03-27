import time
from collections import Counter
import multiprocessing
from functools import partial
import math
from log import get_logger
from origami_greedy import Origami
from origamiprepostprocess import OrigamiPrePostProcess


class ProcessFile(Origami):
    """
    Decoding and encoding will call this class. And this class will call
    the origami method to handle individual origami. This file will also call
    """

    def __init__(self, verbose):
        """
        This will combine all the origami and reconstruct the file
        :param verbose:
        """
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.logger = get_logger(verbose, __name__)
        # Will be updated later during checking number how much redundancy we will need
        self.number_of_bit_per_origami = 29

    def _find_optimum_index_bits(self, bits_needed_to_store):
        """
        Find the optimum number of index bits
        :param bits_needed_to_store:
        :param available_capacity:
        :return:
        """
        total_capacity = self.row * self.column
        checksum_allocation = len(self.get_checksum_relation())
        parity_allocation = len(self.get_parity_relation())
        available_capacity = total_capacity - checksum_allocation - parity_allocation - 4

        for i in range(1, available_capacity):
            capacity_after_index = available_capacity - i
            index_bit_required = math.ceil(bits_needed_to_store / capacity_after_index)
            if 2**i >= index_bit_required:
                return i, capacity_after_index, index_bit_required
        raise Exception("File size is to large to store in the given capacity")

    def encode(self, file_in, file_out, formatted_output=False):
        """
        Encode the file
        :param file_in: File that need to be encoded
        :param file_out: File where output will be saved
        :param formatted_output: Output will written as a matrix
        :return:
        """
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

        bits_needed_to_store = len(data_in_binary)
        index_bits, data_bit, segment_size = self._find_optimum_index_bits(bits_needed_to_store)
        # Divide into origami from datastream
        for origami_index in range(segment_size):
            origami_data = data_in_binary[origami_index * data_bit: (origami_index + 1) * data_bit].ljust(data_bit, '0')
            encoded_stream = self._encode(origami_data, origami_index, data_bit)
            if formatted_output:
                print("Matrix -> " + str(origami_index), file=file_out)
                self.print_matrix(self.data_stream_to_matrix(encoded_stream), in_file=file_out)
            else:
                file_out.write(encoded_stream + '\n')
        file_out.close()
        self.logger.info("Encoding done")
        return segment_size, data_bit

    def single_origami_decode(self, single_origami, ior_file_name, correct_dictionary, common_parity_index,
                              minimum_temporary_weight, maximum_number_of_error, false_positive):
        current_time = time.time()
        self.logger.info("Working on origami(%d): %s", single_origami[0], single_origami[1])
        if len(single_origami[1]) != self.row * self.column:
            self.logger.warning("Data point is missing in the origami")
            return
        try:
            decoded_matrix = super().decode(single_origami[1], common_parity_index, minimum_temporary_weight,
                                            maximum_number_of_error, false_positive)
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
                                decoded_origami=self.matrix_to_data_stream(decoded_matrix['matrix']),
                                decoded_data=decoded_matrix['binary_data'],
                                decoding_time=decoded_time,
                                current_origami_index=single_origami[0]))
            # lock.release()
        return [decoded_matrix, status]

    def decode(self, file_in, file_out, file_size, threshold_data, threshold_parity, maximum_number_of_error,
               individual_origami_info, false_positive, correct_file=False):
        correct_origami = 0
        incorrect_origami = 0
        total_error_fixed = 0
        # Read the file
        try:
            data_file = open(file_in, "r")
            data = data_file.readlines()
            data_file.close()
            # File to store individual origami information
            if individual_origami_info:
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
        index_bits, data_bit, segment_size = self._find_optimum_index_bits(file_size * 8)
        self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = \
            self._matrix_details(data_bit)
        self.data_bit_to_parity_bit = self.get_data_bit_to_parity_bit(self.parity_bit_relation)

        decoded_dictionary = {}
        # If user pass correct file we will create a correct key value pair from that and will compare with our decoded
        # data.
        correct_dictionary = {}
        if correct_file:
            with open(correct_file) as cf:
                for so in cf:
                    ci, cd = self._extract_text_and_index(self.data_stream_to_matrix(so.rstrip("\n")))
                    correct_dictionary[ci] = cd
        # Decoded dictionary with number of occurrence of a single origami
        decoded_dictionary_wno = {}
        origami_data = [(i, single_origami.rstrip("\n")) for i, single_origami in enumerate(data)]
        p_single_origami_decode = partial(self.single_origami_decode, ior_file_name=ior_file_name,
                                          correct_dictionary=
                                          correct_dictionary, common_parity_index=threshold_data,
                                          minimum_temporary_weight=threshold_parity,
                                          maximum_number_of_error=maximum_number_of_error,
                                          false_positive=false_positive)
        # return_value = map(p_single_origami_decode, origami_data)
        optimum_number_of_process = int(math.ceil(multiprocessing.cpu_count()))
        pool = multiprocessing.Pool(processes=optimum_number_of_process)
        return_value = pool.map(p_single_origami_decode, origami_data)
        pool.close()
        pool.join()
        for decoded_matrix in return_value:
            if not decoded_matrix is None and not decoded_matrix[0] is None:
                # Checking status
                if correct_file:
                    if decoded_matrix[1]:
                        correct_origami += 1
                    else:
                        incorrect_origami += 1
                total_error_fixed += int(decoded_matrix[0]['total_probable_error'])
                decoded_dictionary_wno.setdefault(decoded_matrix[0]['index'], []).append(
                    decoded_matrix[0]['binary_data'])

        # perform majority voting
        final_origami_data = [None] * segment_size
        for key, value in decoded_dictionary_wno.items():
            final_origami_data[key] = Counter(value).most_common(1)[0][0]

        missing_origami = [i for i, val in enumerate(final_origami_data) if val is None]
        if len(missing_origami) > 0:
            return -1, incorrect_origami, correct_origami, total_error_fixed, missing_origami
        recovered_binary = "".join(final_origami_data)
        # Remove the padding
        recovered_binary = recovered_binary[:8 * (len(recovered_binary) // 8)]
        with open(file_out, "wb") as result_file:
            for start_index in range(0, len(recovered_binary), 8):
                bin_data = recovered_binary[start_index:start_index + 8]
                # convert bin data into decimal
                decimal = int(''.join(str(i) for i in bin_data), 2)
                if decimal == 0 and start_index + 8 == len(
                        recovered_binary):  # This will remove the padding. If the padding is whole byte.
                    continue
                decimal_byte = bytes([decimal])
                result_file.write(decimal_byte)
        self.logger.info("Number of missing origami :" + str(missing_origami))
        self.logger.info("Total error fixed: " + str(total_error_fixed))
        self.logger.info("File recovery was successfull")
        return 1, incorrect_origami, correct_origami, total_error_fixed, missing_origami


# This is for debugging purpose
if __name__ == '__main__':
    pass
