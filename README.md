# MURA: Time Series Forecasting via Frequency Interpolation

Time series forecasting relies on capturing inherent trend and seasonality of the series. Previous methods such as moving average and learnable decomposition depend only on convolutional kernels, which rely on the local dependency and cannot capture the global variation. In this paper, we introduce a seasonality-trend decomposition technique through Frequency Interpolation Decomposition (FID). By interpolation of complex frequencies that contain global variation, it simultaneously obtains the trend of both the look-back window and the forecast window. Further, we employ an Multi-Layer Perceptron (MLP) to predict the seasonal components (residual frequencies) of the look-back window after performing trend downsampling. Combining FID with residual frequency forecasting forms the lightweight and powerful method, called MURA. Extensive experimental results show that the MURA method achieves state-of-the-art performance while maintaining extremely advanced computational efficiency with only about $3k$ parameters. Moreover, FID is also used to enhance the performance of other methods such as PatchTST and iTransformer.

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

Southeast University, Nanjing, China