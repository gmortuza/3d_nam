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

    def decode(self, file_in, file_out):
        pass


if __name__ == '__main__':
    config_ = Config('config.yaml')
    process_file = ProcessFile(config_)
    process_file.encode('../test.txt', 'encoded.txt')
