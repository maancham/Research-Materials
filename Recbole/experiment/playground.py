import yaml
from pathlib import Path



params_file_name = "BPR.yaml"
model_params_dir = "model params"
params_file_dir = model_params_dir + '/' + params_file_name


with open(Path(params_file_dir)) as params_file:
    config_dict = yaml.full_load(params_file)

print(config_dict)
