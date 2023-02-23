from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

import sys
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


"""
TODO:
- add new files (run model, model names, prediction)
- do we need fine-tuning at this stage?
"""






def RunModel(data_filename, model_name, config_file_list, params_file_name, should_save=False):

    if(params_file_name):
        model_params_dir = "model params"
        params_file_dir = model_params_dir + '/' + params_file_name
        with open(Path(params_file_dir)) as params_file:
            config_dict = yaml.full_load(params_file)
    else:
        config_dict = None

    with open(Path(config_file_list)) as config_file:
            config_file = yaml.full_load(config_file)

    config_dict = {**config_dict, **config_file}

    run_recbole(dataset=data_filename, model=model_name, config_dict = config_dict, saved=should_save)



def ModelHandler(models, dataset):

    dataset_config = dataset + '.yaml'
    for model in models:

        if (model in ['Pop', 'ItemKNN']):
            params_file_name = None
        else:
            params_file_name = model + '.yaml'

        RunModel(dataset, model, dataset_config, params_file_name, should_save=True)




def ExportAlgoRec(userId_list, algo_name, external_item_ids, scores, item_file_path):

    new_dir = 'output/' + algo_name
    output_dir = Path(new_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(item_file_path, encoding='utf-8') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')

        for i in range(len(userId_list)):
            user_rec_list = []

            rec_items = list(external_item_ids[i])
            rec_item_scores = scores[i].tolist()
            rec_item_scores = ['%.2f' % elem for elem in rec_item_scores]

            for row in rd:
                if (row[0] in rec_items):
                    new_row = row
                    new_row.append(rec_item_scores[rec_items.index(row[0])])
                    user_rec_list.append(new_row)
            

            user_recs_df = pd.DataFrame(user_rec_list, columns = ['movieId', 'title', 'year', 'genres', 'prediction'])
            file_name = 'user_' + str(userId_list[i]) + '_' + algo_name + '_recommendations.csv'
            user_recs_df.to_csv(output_dir / file_name)

            fd.seek(0)



def ExportRecsHandler(userId_list, k):

    saved_folder_dir = Path("saved")
    saved_model_names = []
    for item in saved_folder_dir.iterdir():
        saved_model_names.append(f"{item}")

    for saved_model_name in saved_model_names:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=Path(saved_model_name),
        )
        uid_series = dataset.token2id(dataset.uid_field, userId_list)
        topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=k, device=config['device'])
        external_item_ids = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
        algo_name = saved_model_name.split('/')[-1].split('-')[0]
        item_file_path = Path('dataset/' + dataset.dataset_name + '/' + dataset.dataset_name + '.item')
        ExportAlgoRec(userId_list, algo_name, external_item_ids, topk_score, item_file_path)




if __name__ == '__main__':

    if (len(sys.argv) != 2):
        raise ValueError("Must (only) input dataset name!")
    
    dataset_name = sys.argv[1:][0]

    ## NeuMF => NCF
    models = ['Pop', 'ItemKNN', 'BPR', 'DMF', 'NeuMF',
            'NAIS', 'FISM', 'NGCF', 'LightGCN', 'ENMF',
            'CDAE', 'MultiVAE']
    
    models = ['BPR']

    ModelHandler(models, str(dataset_name))

    # userId_list = ['0', '5']

    # ExportRecsHandler(userId_list, 20)



