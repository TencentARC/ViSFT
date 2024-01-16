import torch

checkpoint = torch.load('path_to/EVA02_CLIP_E_psz14_plus_s9B.pt', map_location=torch.device("cpu"))

new_chkpt_dict = {}
for k, v in list(checkpoint.items()):
        if k.startswith('text') or k.startswith('logit'):
                continue
        if k.startswith('visual'):
                if 'head' in k or 'cls_token' in k:
                        continue
                new_key = k.replace('visual.','')
                new_chkpt_dict[new_key] = v
        else:
                new_key = k
                new_chkpt_dict[new_key] = v
        print(new_key, '        ', v.shape)

print("len new_chkpt_dict", len(new_chkpt_dict))
torch.save(new_chkpt_dict, 'path_to/EVA02_CLIP_E_psz14_plus_s9B_Visual.pt')
for k, v in list(checkpoint.items()):
        print(k, '        ', v.shape)
