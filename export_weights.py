import os
import torch

# Example here
#
#
# ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
# DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"
# DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"
# MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"
# VALENTINI_NC = ROOT + 'valentini_nc-93fc4337.th' 

# def _demucs(pretrained, url, **kwargs):
#     model = Demucs(**kwargs, sample_rate=16_000)
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
#
# def master64(pretrained=True):
#     return _demucs(pretrained, MASTER_64_URL, hidden=64)
#
# model = master64()

model = None # Load model here


basedir = 'denoiser'

for name, param in model.named_parameters():
    print("Name: ", name, " Param: ", param.shape)
    filename = str(name) + '.bin'
    path = os.path.join(basedir, filename)
    torch.save(param, path)