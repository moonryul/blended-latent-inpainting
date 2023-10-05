import torch
import argparse
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

##https://github.com/huggingface/diffusers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, #required=True,
                    default="",help="The target text prompt"
    )
    parser.add_argument(
        "--init_image", type=str, #required=True,
           default="inputs/img.png", help="The path to the input image"
    )
    parser.add_argument(
        "--mask", type=str, #required=True,
        default="inputs/mask.png", help="The path to the input mask"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        ##default="stabilityai/stable-diffusion-2-1-base",
        default="stabilityai/stable-diffusion-2-inpainting",
        help="The path to the HuggingFace model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The number of images to generate"
    )
    parser.add_argument(
        "--blending_start_percentage",
        type=float,
        default=0.25,
        help="The diffusion steps percentage to jump",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/res.jpg",
        help="The destination output path",
    )

    args = parser.parse_args()
    return args

#End def parse_args()


def encode_image(image):
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
    image = (image + 1) /2.0
    image = image.half()
    # latents = self.vae.encode(image)["latent_dist"].mean
    # latents = latents * 0.18215 
    #MJ: = latents * scale_factor = latents/std(z) = the normalized values with unit variance

    return image

def encode_mask(mask_path: str, dest_size=(64, 64)):
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask) / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = mask[np.newaxis, np.newaxis, ...]
    mask = torch.from_numpy(mask).half().to("cuda")

    return mask, org_mask
    

args=parse_args()


image = Image.open(args.init_image)
height, width = image.size
image = image.resize((height, width), Image.BILINEAR)
image = np.array(image)[:, :, :3]

#MJ: z_{init} ~ E(x), x= source image: z_{init} = source_latents
image = encode_image(image)
        
#MJ:  m_{latent} = downsample(m): resize the mask to dest_size=(64, 64): m_{latent} = latent_mask
mask, org_mask = encode_mask(args.mask, dest_size=(height,width))

        
pipe = StableDiffusionInpaintPipeline.from_pretrained(
   # "stabilityai/stable-diffusion-2-inpainting",
    args.model_path,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
prompt =""
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is


image = pipe(prompt=prompt, image=image, mask_image=mask).images[0]

#MJ: you can provide additional parameters strength, num_inference_steps,
#         guidance_scale, as shown in the following __call__ method of StableDiffusionInPaintingPipeline:
#         def __call__(
#         self,
#         prompt: Union[str, List[str]] = None,
#         image: PipelineImageInput = None,
#         mask_image: PipelineImageInput = None,
#         masked_image_latents: torch.FloatTensor = None,
#         height: Optional[int] = None,
#         width: Optional[int] = None,
        
#         strength: float = 1.0,  #MJ: stength is mentioned in __call__ of StableDiffusionInpaintPipeline
        
#         num_inference_steps: int = 50,
#         guidance_scale: float = 7.5,
        
        
image.save("./yellow_cat_on_park_bench.png")