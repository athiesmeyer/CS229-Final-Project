import lib
import numpy as np
from pathlib import Path
import pandas as pd

#### NOTE: The generation of corruption patterns for MCAR, MNAR, and MAR is adapted from the jenga package,
#### credit to https://github.com/schelterlabs/jenga

def mask_for_imputing(data, col_to_impute, type, prop):
        data = pd.DataFrame(data.cpu().numpy())     
        n = data.shape[0]
        d = data.shape[1]  
        if type == "MCAR":
            mask = np.random.permutation(n)[:int(n*prop)]
        else:
            n_vals_to_discard = int(n * prop)
            perc_start = np.random.randint(0, n - n_vals_to_discard)
            perc_indices = range(perc_start, perc_start + n_vals_to_discard)
            if type == "MNAR":
                mask = data[col_to_impute].sort_values().iloc[perc_indices].index
            elif type == "MAR":
                depends_on_col = np.random.choice(list(set(range(d)) - {col_to_impute}))
                mask = data[depends_on_col].sort_values().iloc[perc_indices].index
        
        return mask

def save_results(result, parent_dir, col_name, exp_type, exp_prop):
    results_path = Path(parent_dir + "/imp_exp_results")
    results_path.mkdir(exist_ok=True, parents=True)

    save_path = results_path / f"results.json"
    if not save_path.is_file():
        results = {f"{col_name}": {f"{exp_type}" : {f"{exp_prop}" : result}}}
    else:
        results = lib.load_json(save_path)
        if col_name in results.keys():
            if exp_type in results[col_name].keys():
                results[col_name][exp_type][exp_prop] = result
            else:
                results[col_name][exp_type] = {f"{exp_prop}" : result}
        else:
            results[col_name] = {f"{exp_type}" : {f"{exp_prop}" : result}}

    lib.dump_json(results, save_path)