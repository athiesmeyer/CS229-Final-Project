import argparse
import json
import lib
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sklearn.model_selection
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('ds_name', type=str)
args = parser.parse_args()

ds_name = args.ds_name

exp_dir = Path(f"exp/{ds_name}/ddpm_cb_best/config.toml")
pipeline = Path(f"scripts/pipeline.py")

props_missing_data = [0.01, 0.1, 0.3, 0.5]
types = ["MCAR", "MAR"]

for type in types:
    for prop in props_missing_data:
        subprocess.run(['python', f'{pipeline}', '--config', f'{exp_dir}', '--sample_partial', f'--type {type}', f'--proportion {prop}'], check=True)
