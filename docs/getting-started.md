## Getting started with using this repo

1. Clone the repo Int2Int.
2. Install `uv`. It is a package manager for python that we will use for speed.
3. Create a virtual environment at ./.venv with Python 3.11. Since this is the latest supported with PyTorch. For future feel free to use the latest available. `uv venv .venv --python 3.11`
4. Activate the environment `source .venv/bin/activate`
5. Install PyTorch for your specific hardware and OS. For Nvidia with CUDA 12.8 (June 2025) and Linux the pytorch website specifies `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`.

## Running models that generate their own data

- On laptops and local devices a batch size of 256 -1024 is recommended. 
Run `python train.py --operation "invert_f_b_binary" --dump_path ./data/ --exp_name boolean_f --exp_id 07_02_invert_f_b_binary_warm_2e4 --base 2 --batch_size 256 --epoch_size 100000 --max_epoch 200 --optimizer 'adam_warmup,lr=0.0002'`

- On the cloud Nvidia GPUs here is an example with batch size 16384
Run `python train.py --operation invert_f_b_hex --dump_path './data/' --exp_name boolean_f --exp_id 07_02_invert_f_b_hex_warm_2e4 --base 2 --batch_size 16384 --epoch_size 500000 --max_epoch 200 --optimizer 'adam_warmup,lr=0.0002'`

## Train and Test with fixed Data

1. First we generate data `python train.py --operation "boolean_f_hex" --export_data true --epoch_size 500000 --dump_path ./data/ --exp_name boolean_f --exp_id 07_01_hex_train --batch_size 1024 --max_epoch 1`


2. Then we generate test data `python train.py --operation "boolean_f_hex" --export_data true --epoch_size 30000 --dump_path ./data/ --exp_name boolean_f --exp_id 07_01_hex_test --batch_size 1024 --max_epoch 1`

3. Then we copy the file to convenient locations `cp ./data/boolean_f/07_01_hex_train/data.prefix ./data/boolean_f_hex.train` and `cp ./data/boolean_f/07_01_hex_test/data.prefix ./data/boolean_f_hex.test`.

4. Then we train the model and test it `python train.py --operation "boolean_f_hex" --train_data ./data/boolean_f_hex.train --eval_data ./data/boolean_f_hex.test --dump_path ./dump/ --exp_name boolean_f --exp_id 07_01_hex_from_file --max_epoch 1 --validation_metric _valid_arithmetic_xe_loss,valid_arithmetic_acc`

## Tuning

- Change the learning rate and the optimizer by `--optimizer "adam_warmup,lr=0.0002` for example.