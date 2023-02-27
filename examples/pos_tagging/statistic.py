from os.path import join, dirname 
import data

working_folder = dirname(__file__)
dataset_folder = join(working_folder, "vlsp2013")

corpus = data.DataReader.load_tagged_corpus(
    join(working_folder, "tmp/vlsp2013"), train_file="train.txt", test_file="test.txt"
)
pwd = dirname(__file__)
output_dir = join(pwd, "tmp/pos_tag")
