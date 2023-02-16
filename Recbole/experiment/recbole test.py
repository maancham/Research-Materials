from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

import csv
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


"""
TODO:
- add other algorithms
- fine tuning still left?
- export results for multiple algorithms
- add new files (run model, model names, prediction)
"""






def RunModel(data_filename, model_name, config_file_list, params_file_name, should_save=False):

    # if(params_file_name):
    #     model_params_dir = "model params"
    #     params_file_dir = model_params_dir + '/' + params_file_name
    #     with open(Path(params_file_dir)) as params_file:
    #         config_dict = yaml.full_load(params_file)
    # else:
    #     config_dict = None

    config_dict = None
    
    run_recbole(dataset=data_filename, model=model_name, config_file_list=[config_file_list],
                config_dict = config_dict, saved=should_save)





def ExportRecs(userId_list, external_item_ids, scores, item_file_path):


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
            file_name = 'user_' + str(userId_list[i]) + '_recommendations.csv'
            user_recs_df.to_csv(file_name)




if __name__ == '__main__':

    # config = Config(model='BPR', dataset='ml-small', config_file_list=['ml-small.yaml'])
    # dataset = create_dataset(config)
    # train_data, valid_data, test_data = data_preparation(config, dataset)


    ## FM => SVD++
    ## NeuMF => NCF
    models = ['Pop', 'ItemKNN', 'BPR', 'FM', 'NeuMF']

    for model in models:

        if (model in ['Pop', 'ItemKNN']):
            params_file_name = None
        else:
            params_file_name = model + '.yaml'

        RunModel('ml-small', model, 'ml-small.yaml', params_file_name, should_save=True)





    # config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    #     model_file='saved\BPR-Feb-13-2023_14-18-04.pth',
    # )

    # userId_list = ['0']

    # uid_series = dataset.token2id(dataset.uid_field, userId_list)

    # topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=20, device=config['device'])
    # external_item_ids = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())


    # ExportRecs(userId_list, external_item_ids, topk_score, Path("ml-small/ml-small.item"))
    

