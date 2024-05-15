import os
from core.models.model_module_infer import model_module
from PIL import Image
import torch
import torch.nn as nn
from core.common.utils import remove_duplicate_word
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(torch.cuda.device_count())

model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_video_diffuser_8frames.pth']
inference_tester = model_module(data_dir='checkpoints/', pth=model_load_paths,
                                fp16=False, load_weights=True)  # turn on fp16=True if loading fp16 weights
inference_tester = inference_tester.to(device)
inference_tester.eval()
a = nn.DataParallel(inference_tester, device_ids=[0])

prompt = "A beautiful oil painting of a birch tree standing in a spring meadow with pink flowers, a distant mountain towers over the field in the distance. Artwork by Alena Aenami"
"""
# prompt = "Create a vibrant and whimsical scene featuring three best friends: Poppy the playful monkey, Bamboo the gentle panda, and Pini the joyful child. The setting is a lush, enchanted forest with colorful flowers, tall trees, and a sparkling stream. Poppy swings from a tree branch, Bamboo lounges contentedly on the grass, and Pini dances nearby, all surrounded by a magical, fairy-tale atmosphere. Include bright, cheerful colors and an overall sense of friendship and adventure."

# Generate image
images = a(xtype=['image'],
           condition=[prompt],
           condition_types=['text'],
           n_samples=1,
           image_size=512,
           ddim_steps=500)

i = images[0][0].cpu().numpy()
i = i.transpose(1, 2, 0)

img = Image.fromarray((i * 255).astype('uint8'))
img.save('image.png')
"""
i = Image.open('image.png')
i = np.array(i)
i = torch.Tensor(i).permute(2, 0, 1).unsqueeze(0).to(device)
print(i.shape)
"""
text = a(xtype=['text'],
         condition=[i],
         condition_types=['image'],
         n_samples=4,
         ddim_steps=100,
         scale=7.5, )

sentenses = text
output = []
for out in sentenses[0]:
    text = a.module.net.optimus.tokenizer_decoder.decode(out.tolist(), clean_up_tokenization_spaces=True)
    text = text.split()[1:-1]
    text = ' '.join(text)
    output.append(text)

x = output
xnew = []
for xi in x:
    xi_split = xi.split()
    xinew = []
    for idxi, wi in enumerate(xi_split):
        if idxi!=0 and wi==xi_split[idxi-1]:
            continue
        xinew.append(wi)
    xnew.append(remove_duplicate_word(' '.join(xinew)))
x = xnew

print(output)
print(x)
"""

images = a(['image'],
           condition=[i, prompt],
           condition_types=['image', 'text'],
           n_samples=1,
           image_size=512,
           mix_weight={'image': 1, 'text': 1},
           ddim_steps=100,)

images[0][0]

i = images[0][0].cpu().numpy()
i = i.transpose(1, 2, 0)

img = Image.fromarray((i * 255).astype('uint8'))
img.save('image.png')
