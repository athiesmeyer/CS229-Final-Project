# CS229-Final-Project

Guidelines to Run Code:
Follow directions at https://github.com/rotot0/tab-ddpm, except use 
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
instead of cu111 version of cuda (couldn't get it to work).

Again, follow their instructions to download the datasets (on Windows I manually extracted the data tar ball).

If on Windows, copy the data, lib, tab_ddpm folders into the script folder (necessary for me to prevent failed import statements, didn't have the issue on Mac).

I focused on running the line 
python scripts/pipeline.py --config exp/churn2/ddpm_cb_best/config.toml --train
and have not tested other commands (this experiment ended up taking my laptop ~2 and a half hours to run)

For this command specifically, navigate to exp/churn2/ddpm_cb_best/config.toml and change device to whatever you need (if you don't have an NVIDIA gpu, I think you have to change to CPU. I had to change mine from "cuda:1" to "cuda:0" for it to work because my GPU only has one thread)

Finally, you may get an error message about the type of tensor used for indexing. I believe I navigated to where the bug threw the error and just added .long() methods but I can't find where. Let me know if you get this error.

## Run a single experiment and compare results to mean/mode
python scripts/pipeline.py --config exp/abalone/ddpm_cb_best/config.toml --sample_partial --to_impute Length --exp_type MCAR --exp_prop 0.1 --compare

## Run all experiments and compare results to mean/mode
python scripts/run_exps.py abalone Length

(Note: if using a provided dataset besides abalone, the --to_impute column is named "impute". Col names can be changed in the info.json file in the data folder. Also, the save results function is a bit wonky and may throw an error if previous results already exist, feel free to improve)
