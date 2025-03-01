# MURA: Time Series Forecasting via Frequency Interpolation

## Requirements
We recommend using the latest versions of dependencies. However, you can refer to the requirements.txt file to set up the same environment as we used.

## Datasets

Download: https://github.com/thuml/Autoformer

Please place the datasets in the ./datasets directory.


## Usage
All experiments can be reproduced using the scripts directory.

You need to modify the `data_path` configuration in file **configs/forecast/long.yaml** to the correct path of your dataset.

### Train

Version: MURA_base
```
bash scripts/max/ECL.sh
```
Version: MURA_MIN
```
bash scripts/min/ETTh1.sh
```

### Test

You need to modify the `ckpt_path` configuration in file **configs/eval.yaml** to the correct path of your ckpt.

Next, you need to modify `train.py` to `eval.py` in the `.sh` file.
```
HYDRA_FULL_ERROR=1 python src/train.py \ # change train.py to eval.py
```
## Citation

## License
This repo is licensed under the MIT License - see the LICENSE file for details.