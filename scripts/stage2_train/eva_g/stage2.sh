# you may need to adapt the following line to your cluster setting
# srun --mem=300g --nodes=8 --gres=gpu:8 --time=4300 --cpus-per-task=40 \

export PYTHONPATH=${PYTHONPATH}:.

python mmf_cli/run.py config=projects/visft/configs/stage2/eva_g/stage2.yaml \
    datasets=caption_coco,detection_coco,segment_coco \
    model=visft run_type=train \
    env.save_dir=./save/visft/stage2_train/eva_g/visft_lora \
    training.batch_size=8 \
    checkpoint.resume=True