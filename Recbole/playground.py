import yaml
from pathlib import Path



params_file_name = "BPR.yaml"
model_params_dir = "model params"
params_file_dir = model_params_dir + '/' + params_file_name


with open(Path(params_file_dir)) as params_file:
    config_dict = yaml.full_load(params_file)

# print(config_dict)

# print(Path('saved\BPR-Feb-16-2023_20-27-54.pth'))


saved_folder = Path("saved")
for item in saved_folder.iterdir():
    # print(type(f"{item}"))
    a = f"{item}"