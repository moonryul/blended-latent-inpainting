import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

import sys
print('sys.path', sys.path)
sys.path.append('.')
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  #MJ: (1,H,W,C) => (1,C,H,W)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    #MJ: Convert the mask into the binary one
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1 - mask) * image #MJ: the mask region becomes zero

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                
                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                #MJ: The masked_image is put to the cond_stage_model?
                #It is because of     cond_stage_config: __is_first_stage__ in config
                c = model.cond_stage_model.encode(batch["masked_image"]) #MJ: encode into the 1/8 space
                
                #MJ: 
                cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:]) #MJ: size=(H,W) of the masked_image which is in the latent space
                
                c = torch.cat((c, cc), dim=1) #MJ: c= the stack of the masked_image and the mask

                shape = (c.shape[1] - 1,) + c.shape[2:] #MJ: c=(B,3+1,H,W): shape=(3,H,W)= the shape of the image
                #MJ: I modified omri's sampler.sample() call, by providing mask, org_mask, and init_image as additional parameters
                samples_ddim, _ = sampler.sample(
                    S=opt.steps,
                    
                    conditioning=c,
                    
                    batch_size=c.shape[0],
                    shape=shape,
                    mask=c,
                    org_mask=batch["mask"], 
                    init_image=image,
                    percentage_of_pixel_blending = 0.1,
                
                    verbose=False,
                    
                    
                )
           
    # def sample(
    #     self,
    #     S,
    #     batch_size,
    #     shape,
    #     conditioning=None,
    #     callback=None,
    #     normals_sequence=None,
    #     img_callback=None,
    #     quantize_x0=False,
    #     eta=0.0,
    #     mask=None,            #MJ: provide mask (latent_mask)
    #     org_mask=None,        #MJ: provide mask in the original image
    #     x0=None,
    #     temperature=1.0,
    #     noise_dropout=0.0,
    #     score_corrector=None,
    #     corrector_kwargs=None,
    #     verbose=True,
    #     x_T=None,
    #     log_every_t=100,
    #     unconditional_guidance_scale=1.0,
    #     unconditional_conditioning=None, #MJ: Aha! When we use the stable diffusion for inpainting, we have an option to set unconditional_conditioning to None
    #     skip_steps=0,
    #     init_image=None,
    #     percentage_of_pixel_blending=0, #MJ: provide percentage_of_pixel_blending = 0.1
    #     # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
    #     **kwargs,
    # ):
        
        
                    
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
                predicted_image = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                ) #MJ: predicted_image is the generated image for the mask region

                inpainted = (1 - mask) * image + mask * predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
