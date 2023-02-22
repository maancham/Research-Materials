import yaml
from pathlib import Path
import pandas as pd
import csv



params_file_name = "BPR.yaml"
model_params_dir = "model params"
params_file_dir = model_params_dir + '/' + params_file_name



# print(config_dict)

# print(Path('saved\BPR-Feb-16-2023_20-27-54.pth'))


# new_dir = 'output/user_' + '0' 
# output_dir = Path(new_dir)
# output_dir.mkdir(parents=True, exist_ok=True)

# user_recs_df = pd.DataFrame([1,2,3], [4,5,6])
# file_name = 'user_0' + '_' + 'BPR' + '_recommendations.csv'
# user_recs_df.to_csv(output_dir / file_name)

