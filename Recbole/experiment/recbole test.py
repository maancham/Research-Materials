from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

# run_recbole(model='BPR', dataset='ml-100k')



if __name__ == '__main__':
    config = Config(model='BPR', dataset='ml-small', config_file_list=['ml-small.yaml'])
    dataset = create_dataset(config)

    # print(dataset)


    run_recbole(dataset='ml-small', model='BPR', config_file_list=['ml-small.yaml'])
    # train_data, valid_data, test_data = data_preparation(config, dataset)