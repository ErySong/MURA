model=MURA
data=Solar
L=720
trend_freq=0.95
batch_size=512
lr=0.01
d_model=256
stride=48
save_dir=./logs_MAX/$model/$data
for H in 96; do
    HYDRA_FULL_ERROR=1 python src/train.py \
        model=${model} \
        data=${data} \
        trainer=gpu1 \
        logger.save_dir=${save_dir} \
        trainer.max_epochs=40 \
        forecast.seq_len=$L \
        forecast.pred_len=$H \
        forecast.batch_size=$batch_size \
        forecast.lr=$lr \
        model.net_params.stride=12 \
        model.net_params.d_model=512 \
        model.net_params.norm_type=seq \
        model.net_params.trend_freq=$trend_freq \
        model.net_params.visual_cs=0
done
for H in 192; do
    HYDRA_FULL_ERROR=1 python src/train.py \
        model=${model} \
        data=${data} \
        trainer=gpu1 \
        logger.save_dir=${save_dir} \
        trainer.max_epochs=40 \
        forecast.seq_len=$L \
        forecast.pred_len=$H \
        forecast.batch_size=$batch_size \
        forecast.lr=$lr \
        model.net_params.stride=$stride \
        model.net_params.d_model=1024 \
        model.net_params.norm_type=seq \
        model.net_params.trend_freq=$trend_freq \
        model.net_params.visual_cs=0
done
for H in 336; do
    HYDRA_FULL_ERROR=1 python src/train.py \
        model=${model} \
        data=${data} \
        trainer=gpu1 \
        logger.save_dir=${save_dir} \
        trainer.max_epochs=40 \
        forecast.seq_len=$L \
        forecast.pred_len=$H \
        forecast.batch_size=$batch_size \
        forecast.lr=$lr \
        model.net_params.stride=$stride \
        model.net_params.d_model=$d_model \
        model.net_params.norm_type=seq \
        model.net_params.trend_freq=0.5 \
        model.net_params.visual_cs=0
done
for H in 336; do
    HYDRA_FULL_ERROR=1 python src/train.py \
        model=${model} \
        data=${data} \
        trainer=gpu1 \
        logger.save_dir=${save_dir} \
        trainer.max_epochs=40 \
        forecast.seq_len=$L \
        forecast.pred_len=$H \
        forecast.batch_size=$batch_size \
        forecast.lr=$lr \
        model.net_params.stride=24 \
        model.net_params.d_model=$d_model \
        model.net_params.norm_type=seq \
        model.net_params.trend_freq=$trend_freq \
        model.net_params.visual_cs=0
done