# you may need to adapt the following line to your cluster setting
# srun --mem=300g --nodes=8 --gres=gpu:8 --time=4300 --cpus-per-task=40 \

export PYTHONPATH=${PYTHONPATH}:.

python mmf_cli/run.py config=projects/visft/configs/stage1/eva_g/caption.yaml \
    datasets=caption_coco \
    model=visft run_type=train \
    env.save_dir=./save/visft/stage1_train/eva_g/caption_heads \
    training.batch_size=256 \
    checkpoint.resume=True