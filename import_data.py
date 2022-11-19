import argparse
import json
import lib
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sklearn.model_selection

parser = argparse.ArgumentParser()

parser.add_argument('path', type=str)       # Data should be stored in a .csv file
parser.add_argument('ds_name', type=str)
parser.add_argument('task_type', type=str)  # Task type is either "regression", "binclass", or "multiclass"
parser.add_argument('target', type=str)     # Name in dataset of target variable for task
parser.add_argument('to_impute', type=str)
parser.add_argument('--cat_features',       # List of names in dataset of categorical features
                    nargs = "*",
                    type = str,
                    default = [])
parser.add_argument('--to_drop',            # List of variable names in dataset to remove. NOTE: If you use the arff to csv converter I use, you need to drop "id"
                    nargs = "*",
                    type = str,
                    default = [])

args = parser.parse_args()
path_to_data = Path(args.path)
ds_name = args.ds_name
task_type = args.task_type
target = args.target
to_impute = args.to_impute
cat_features = args.cat_features
to_drop = args.to_drop

out_path_dir = Path(f"data/{ds_name}")
out_path_dir.mkdir(exist_ok=True, parents=True)

data = pd.read_csv(path_to_data)
data = data.drop(to_drop, axis = 1)
# Since we lose column name information downstream, we enforce here that the column to be imputed is the first
# column in the data matrix
data = pd.concat([data[to_impute], data.drop(to_impute, axis=1)],
                 axis=1)

# Split data into train, val, and test sets, with ratios 0.64, 0.16, 0.20 respectively.
# For each, separate numerical, categorical, and target features
np.random.seed(5)

data_train, data_test = sklearn.model_selection.train_test_split(data, test_size=0.36)
data_val, data_test = sklearn.model_selection.train_test_split(data_test, test_size=(20/36))

X_train = data_train.drop(target, axis = 1)
X_val = data_val.drop(target, axis = 1)
X_test = data_test.drop(target, axis = 1)

X_num_train = X_train.drop(cat_features, axis = 1).to_numpy()
np.save(os.path.join(out_path_dir, "X_num_train"), X_num_train)
X_num_val = X_val.drop(cat_features, axis = 1).to_numpy()
np.save(os.path.join(out_path_dir, "X_num_val"), X_num_val)
X_num_test = X_test.drop(cat_features, axis = 1).to_numpy()
np.save(os.path.join(out_path_dir, "X_num_test"), X_num_test)

y_train = data_train[target].to_numpy()
np.save(os.path.join(out_path_dir, "y_train"), y_train)
y_val = data_val[target].to_numpy()
np.save(os.path.join(out_path_dir, "y_val"), y_val)
y_test = data_test[target].to_numpy()
np.save(os.path.join(out_path_dir, "y_test"), y_test)

if cat_features != []:
    X_cat_train = X_train[cat_features].to_numpy()
    np.save(os.path.join(out_path_dir, "X_cat_train"), X_cat_train)
    X_cat_val = X_val[cat_features].to_numpy()
    np.save(os.path.join(out_path_dir, "X_cat_val"), X_cat_val)
    X_cat_test = X_test[cat_features].to_numpy()
    np.save(os.path.join(out_path_dir, "X_cat_test"), X_cat_test)
    impute_cat = set(data_train[to_impute]) == set(X_cat_train[:, 0])
else:
    impute_cat = False

# Create info.json file for dataset containing basic data info

info_dictionary = {"name": ds_name,
                   "id": f"{ds_name}--id",
                   "task_type": task_type,
                   "n_num_features": X_num_train.shape[1],
                   "n_cat_features": X_cat_train.shape[1] if cat_features != [] else 0,
                   "train_size": X_num_train.shape[0],
                   "val_size": X_num_val.shape[0],
                   "test_size": X_num_test.shape[0],
                   "impute_cat": impute_cat}

info_json = json.dumps(info_dictionary, indent = 4)
with open(os.path.join(out_path_dir, "info.json"), "w") as out:
    out.write(info_json)

# Create initial base config file for experiments. We only need to change a few parameters
# for each new base config file.

config = lib.load_config(Path("exp/adult/config.toml"))
config["parent_dir"] = f"exp/{ds_name}"
config["real_data_path"] = str(out_path_dir)
config["num_numerical_features"] = X_num_train.shape[1]
if task_type in ["binclass", "multiclass"]:
    config["model_params"]["is_y_cond"] = True
    config["model_params"]["num_classes"] = len(np.unique(y_train))
else:
    config["model_params"]["is_y_cond"] = False
    config["model_params"]["num_classes"] = 0

config_dir = Path(f"exp/{ds_name}")
config_dir.mkdir(exist_ok = True, parents = True)

lib.dump_config(config, config_dir / "config.toml")
