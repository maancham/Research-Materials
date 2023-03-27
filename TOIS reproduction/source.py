from recbole.quick_start import run_recbole
import sys
import pandas as pd
from pathlib import Path
import yaml




def ModelHandler(models):

    dataset_config =  'ml-1m.yaml'
    model_dict = {}

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
        result = run_recbole(dataset='ml-1m', model=model, config_dict = config_dict, saved=False)

        model_result_list = []
        if (result['best_valid_result']):
            model_result_list.append(result['best_valid_result']['ndcg@10'])
        else:
            model_result_list.append('None')
        model_result_list.append(result['test_result']['ndcg@10'])


        model_dict[model] = model_result_list
        
    df = pd.DataFrame.from_dict(model_dict, orient='index')
    df.columns = ['valid_ndcg', 'test_ndcg']
    return df


if __name__ == '__main__':

    models = ['Pop', 'ItemKNN', 'BPR', 'NeuMF', 'CDAE', 'ENMF',
              'FISM', 'LightGCN', 'MultiVAE', 'NAIS', 'NGCF']
    
    result_df = ModelHandler(models)
    print(result_df)