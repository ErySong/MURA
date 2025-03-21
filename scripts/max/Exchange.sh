model=MURA
data=Exchange
save_dir=./logs_MAX/$model/$data
L=96
trend_freq=0.5
batch_size=256
lr=0.005
d_model=128
stride=1
for H in 96; do
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
        model.net_params.norm_type=no \
        model.net_params.trend_freq=$trend_freq \
        model.net_params.visual_cs=0
done
for H in 192; do
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
        model.net_params.norm_type=no \
        model.net_params.trend_freq=$trend_freq \
        model.net_params.visual_cs=0
done
for H in 336; do
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
        model.net_params.norm_type=no \
        model.net_params.trend_freq=$trend_freq \
        model.net_params.visual_cs=0
done
for H in 720; do
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
        model.net_params.stride=12 \
        model.net_params.d_model=512 \
        model.net_params.norm_type=no \
        model.net_params.trend_freq=0.95 \
        model.net_params.visual_cs=0
done