model=MURA
data=ETTh1
L=336
trend_freq=0
batch_size=512
lr=0.01
d_model=128
stride=24
save_dir=./logs_MIN/$model/$data
for H in 96 192 336 720; do
    HYDRA_FULL_ERROR=1 python src/train.py \
        model=${model} \
        data=${data} \
        trainer=gpu0 \
        logger.save_dir=${save_dir} \
        trainer.max_epochs=40 \
        forecast.seq_len=$L \
        forecast.pred_len=$H \
        forecast.batch_size=$batch_size \
        forecast.lr=$lr \
        model.net_params.stride=$stride \
        model.net_params.d_model=$d_model \
        model.net_params.norm_type=seq \
        model.net_params.trend_freq=$trend_freq \
        model.net_params.visual_cs=0
done
