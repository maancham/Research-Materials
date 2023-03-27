from recbole.quick_start import run_recbole
import sys
from pathlib import Path
import yaml




def ModelHandler(models):

    dataset_config =  'ml-1m.yaml'
    for model in models:

        if (model in ['Pop', 'ItemKNN']):
            params_file_name = None
        else:
            params_file_name = model + '.yaml'
        
        if(params_file_name):
            try:
                with open(Path(params_file_name)) as params_file:
                    config_dict = yaml.full_load(params_file)
            except:
                config_dict = {}
        else:
            config_dict = {}

        with open(Path(dataset_config)) as config_file:
                config_file = yaml.full_load(config_file)

        config_dict = {**config_dict, **config_file}
        run_recbole(dataset='ml-1m', model=model, config_dict = config_dict, saved=False)






if __name__ == '__main__':

    models = ['Pop', 'ItemKNN', 'BPR', 'NeuMF', 'CDAE', 'ENMF',
              'FISM', 'LightGCN', 'MultiVAE', 'NAIS', 'NGCF']
    
    ModelHandler(models)