from recbole.quick_start import run_recbole


import sys
from pathlib import Path
import yaml


"""
TODO:
- 
"""


def RunModel(data_filename, model_name, config_file_list, params_file_name, should_save=False):

    if(params_file_name):
        try:
            model_params_dir = "model params"
            params_file_dir = model_params_dir + '/' + params_file_name
            with open(Path(params_file_dir)) as params_file:
                config_dict = yaml.full_load(params_file)
        except:
            config_dict = {}
    else:
        config_dict = {}

    with open(Path(config_file_list)) as config_file:
            config_file = yaml.full_load(config_file)

    config_dict = {**config_dict, **config_file}
    if(model_name in ['GRU4Rec', 'SASRec']):
        config_dict['train_neg_sample_args'] = None
        config_dict['eval_args']['order'] = 'TO'

    run_recbole(dataset=data_filename, model=model_name, config_dict = config_dict, saved=should_save)



def ModelHandler(models, dataset):

    dataset_config = dataset + '.yaml'
    for model in models:

        if (model in ['Pop', 'ItemKNN']):
            params_file_name = None
        else:
            params_file_name = model + '.yaml'

        RunModel(dataset, model, dataset_config, params_file_name, should_save=True)





if __name__ == '__main__':

    if (len(sys.argv) != 2):
        raise ValueError("Args needed: dataset name")
    
    dataset_name = sys.argv[1:][0]


    final_models = ['Pop', 'ItemKNN', 'BPR', 'NeuMF',
                    'CDAE', 'GRU4Rec', 'SASRec', 'MultiVAE', 
                    'EASE', 'ADMMSLIM', 'ConvNCF', 'LINE']
    
    ModelHandler(final_models, str(dataset_name))
