model=MURA
data=M5
L=96
trend_freq=0.1
batch_size=64
lr=0.01
d_model=128
save_dir=./logs_MIN/$model/$data
for H in 24 36 48 60; do
    HYDRA_FULL_ERROR=1 python src/train.py \
        model=${model} \
        data=${data} \
        trainer=gpu1 \
        logger.save_dir=$save_dir \
        trainer.max_epochs=40 \
        forecast.seq_len=$L \
        forecast.pred_len=$H \
        forecast.batch_size=$batch_size \
        forecast.lr=$lr \
        model.net_params.stride=1 \
        model.net_params.d_model=128 \
        model.net_params.norm_type=seq \
        model.net_params.trend_freq=$trend_freq \
        model.net_params.visual_cs=0
done