import torch

model = 'eva_e'
iters = 500
head_lora_joint_ckpt = torch.load('./save/visft/stage2_train/' + model + '/visft_lora/models/model_' + str(iters) + '.ckpt', map_location=torch.device("cpu"))['model']

lora_ckpt_dict = {}

for key, value in list(head_lora_joint_ckpt.items()):
    if 'lora' in key:
        new_key = key.replace("visft_heads.backbone.0","visual")
        print(new_key, '       ', value.shape)
        lora_ckpt_dict[new_key] = value
save_path = './save/visft/stage2_train/' + model + '/visft_lora/models/'
torch.save(lora_ckpt_dict, save_path + model + '_lora_' + str(iters) + '.pt')


