from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

import csv
import pandas as pd

"""
TODO:
- make the paths work for any OS.
- add predicted rating column for each recommendation.
- add algorithm fine-tuning
"""




def BPR(data_filename, config_file_list):
    run_recbole(dataset=data_filename, model='BPR', config_file_list=[config_file_list])




def ExportRecs(user_id, external_item_ids, item_file_path):

    user_rec_list = []

    with open(item_file_path, encoding='utf-8') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if (row[0] in external_item_ids):
                user_rec_list.append(row)

    user_recs_df = pd.DataFrame(user_rec_list, columns = ['movieId', 'title', 'year', 'genres'])
    file_name = 'user_' + str(user_id) + '_recommendations.csv'
    user_recs_df.to_csv(file_name)




if __name__ == '__main__':

    # config = Config(model='BPR', dataset='ml-small', config_file_list=['ml-small.yaml'])
    # dataset = create_dataset(config)
    # train_data, valid_data, test_data = data_preparation(config, dataset)
    # BPR('ml-small', 'ml-small.yaml')

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file='saved\BPR-Feb-13-2023_14-18-04.pth',
    )

    uid_series = dataset.token2id(dataset.uid_field, ['0'])

    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=20, device=config['device'])
    external_item_ids = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())

    
    ExportRecs(0, external_item_ids, 'ml-small\ml-small.item')
    

