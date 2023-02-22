# T2I-Adapter-for-Diffusers
Transfer the T2I-Adapter with any basemodel in diffusers🔥

<img src="https://github.com/TencentARC/T2I-Adapter/raw/main/assets/overview1.png" width="100%" height="100%">

T2I-Adapter, a simple and small (~70M parameters, ~300M storage space) network that can provide extra guidance to pre-trained text-to-image models while freezing the original large text-to-image models. This repository provides the simplest tutorial code for using [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) with diverse basemodel in the diffuser framework. It is very similar to [ControlNet](https://github.com/lllyasviel/ControlNet). You can find the usage of ControlNet in diffusers framework in [this tutorial](https://github.com/haofanwang/ControlNet-for-Diffusers).

# T2I-Adapter + Stable-Diffusion-1.5

As T2I-Adapter only trains adapter layers and keep all stable-diffusion models frozen, it is flexible to use any stable diffusion models as base. Here, I just use [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as an example.

### Download adapter weights
```
mkdir models && cd models
wget https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_keypose_sd14v1.pth
wget https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_seg_sd14v1.pth
wget https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_sketch_sd14v1.pth
cd ..
```

### Install packages
```
# please use this dev version of diffusers, as it has supported new pipeline
git clone https://github.com/HimariO/diffusers-t2i-adapter.git
git checkout general-adapter
cd diffusers-t2i-adapter

# manually change ./src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_adapter.py
# in StableDiffusionAdapterPipeline, comment out following lines
# adapter: Adapter,
# self.register_modules(adapter=adapter)

# then install from source
pip install .
cd ..
```

### Load adapter weight
```
import torch
from typing import *
from diffusers.utils import load_image
from diffusers import StableDiffusionAdapterPipeline, Adapter

model_name = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionAdapterPipeline.from_pretrained(model_name, torch_dtype=torch.float32).to('cuda')

adapter_ckpt = "./models/t2iadapter_seg_sd14v1.pth"
pipe.adapter = Adapter(cin=int(3*64), 
                       channels=[320, 640, 1280, 1280][:4], 
                       nums_rb=2, 
                       ksize=1, 
                       sk=True, 
                       use_conv=False)
pipe.adapter.load_state_dict(torch.load(adapter_ckpt))
pipe.adapter = pipe.adapter.to('cuda')
```

### Inference
```
@torch.no_grad()
def get_color_masks(image: torch.Tensor) -> Dict[Tuple[int], torch.Tensor]:
    h, w, c = image.shape
    assert c == 3
    
    img_2d = image.view((-1, 3))
    colors, freqs = torch.unique(img_2d, return_counts=True, dim=0)
    colors = colors[freqs >= h]
    color2mask = {}
    for color in colors:
        mask = (image == color).float().max(dim=-1).values
        color = color.cpu().numpy().tolist()
        color2mask[tuple(color)] = mask
    return color2mask
    
mask = load_image("./diffusers-t2i-adapter/motor.png")

prompt = ["A black Honda motorcycle parked in front of a garage"]

image = pipe(prompt, [mask, mask]).images[0]
image.save('test.jpg')
```

You can get the results as below, input segmentation image (left), text-guided generated results (right).

<img src="https://raw.githubusercontent.com/HimariO/diffusers-t2i-adapter/general-adapter/motor.png" width="35%" height="35%"> <img src="https://github.com/haofanwang/T2I-Adapter-for-Diffusers/blob/main/test.jpg" width="35%" height="35%">

If you want to use pose as input,

```
mask = load_image("./diffusers-t2i-adapter/pose.png")

prompt = ["A gril"]

# note the difference here!
image = pipe(prompt, [mask]).images[0]
image.save('result.jpg')
```

Please note that it is required to use correct pose format to make sure the generated results are satisfied. For the pre-trained T2I-Adapter (pose), you need to use COCO format pose. [MMPose](https://github.com/open-mmlab/mmpose) is recommendated. The following examples show the difference, OpenPose format (upper), COCO format (bottom)

<img src="https://github.com/haofanwang/T2I-Adapter-for-Diffusers/blob/main/pose3.png" width="35%" height="35%"> <img src="https://github.com/haofanwang/T2I-Adapter-for-Diffusers/blob/main/pose3_res.jpg" width="35%" height="35%">

<img src="https://github.com/haofanwang/T2I-Adapter-for-Diffusers/blob/main/pose2.png" width="35%" height="35%"> <img src="https://github.com/haofanwang/T2I-Adapter-for-Diffusers/blob/main/pose2_res.jpg" width="35%" height="35%">


# T2I-Adapter + Stable-Diffusion-1.5 + Inpainting
Coming soon!

# Acknowledgement
The diffusers pipeline is supported by [HimariO](https://github.com/HimariO/diffusers-t2i-adapter/blob/adapter/main.py), this repo is highly built on the top of it (fixed several typos) and works just as a handy tutorial.
