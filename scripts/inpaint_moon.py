import argparse, os, sys, glob
from omegaconf import OmegaConf
from einops import repeat
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

import sys

sys.path.append('.')
print('sys.path', sys.path)
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
        batch[k] = batch[k] * 2.0 - 1.0 #MJ: [0,1] => [-1,1]; but mask should not be [-1,1]? => It is OK as condition to Unet.
        #MJ: we will also use the usual mask as well.
    return batch


def make_batch_sd(
        image_path,
        mask_path,
        device,
        num_samples=1):
    image = np.array(Image.open(image_path).convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    #MJ: mask: white in the mask region, black in the background
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5) #MJ: masked_image = background

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
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
    #MJ: model: 
    model.load_state_dict(
        torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    num_samples =1
    
    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image_path, mask_path in tqdm(zip(images, masks)): 
                outpath = os.path.join(opt.outdir, os.path.split(image_path)[1]) #MJ: os.path.split(image) = ('inputs', 'dog.png')
                
                #MJ: batch = make_batch(image_path, mask_path, device=device) #MJ: batch = {"image": image, "mask": mask, "masked_image": masked_image}
                batch = make_batch_sd(image_path, mask_path, device=device,num_samples=1)
                # encode masked image and concat downsampled mask
                #MJ: The masked_image is put to the cond_stage_model?
                #It is because of     cond_stage_config: __is_first_stage__ in config
                #c = model.cond_stage_model.encode(batch["masked_image"]) #MJ: batch["masked_image"]=torch.Size([1, 3, 512, 512]): encode into the 1/8 space
                #MJ: => VQModelInterface.encode( batch["masked_image"]); first_stage_config: target: ldm.models.autoencoder.VQModelInterface
                #MJ:     cond_stage_config: __is_first_stage__: cond_stage_model is also the first_stage_model
                
                
                #MJ: create the original image
                init_image = batch['image'] #MJ: init_image in [-1,1] has the batch dim like all other tensors
                b,c,h,w= init_image.shape  #MJ: (1,3,512,512)
                org_mask = batch['mask']   #MJ: in [0,1]
                assert b==num_samples, "b and num_samples should be equal"
                                
                c_cat = list()
                # for ck in model.concat_keys: #MJ: concat_keys=("mask", "masked_image"), masked_image_key="masked_image",
                for ck in ("mask", "masked_image"):        
                    #Downsample the mask and the masked_image    
                    cc = batch[ck].float()
                    # if ck != model.masked_image_key: #MJ: ck="mask"
                    if ck != "masked_image": #MJ: "mask"
                        #bchw = [num_samples, 4, h // 8, w // 8]
                        bchw = [num_samples, 1, h // 4, w // 4]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])  #MJ: cc = (1,1,128,128)
                        latent_mask = cc #MJ: cc is an ordinary tensor (b,c,h,w)
                    else: #MJ: ck = "masked_image": use the encoder of vae
                        cc = model.get_first_stage_encoding(
                            model.encode_first_stage(cc))   #MJ: cc = (1,3,128,128)
                    c_cat.append(cc)
                 #End for ck in model.concat_keys: #MJ: concat_keys=("mask", "masked_image") 
                 #c_cat = [mask, masked_image]   
        
                c_cat = torch.cat( c_cat, dim=1) #MJ: c_cat: (1,4,128,128)
           
             #MJ: I modified omri's sampler.sample() call, by providing mask, org_mask, and init_image as additional parameters
            #  def sample(self,
            #    S,
            #    batch_size,
            #    shape,
            #    conditioning=None,
            
                shape = [model.channels, h // 4, w // 4] #MJ: model.channels =4 from config
                
                samples_ddim, _ = sampler.sample(  #MJ: sampler.sample() is decorated by @torch.no_grad()
                    S=opt.steps,
                    
                    conditioning=c_cat,
                    
                    batch_size=num_samples,
                    shape=shape,
                    mask=latent_mask,
                    org_mask=org_mask, 
                    init_image=init_image,
                    percentage_of_pixel_blending = 1/50.0,
                
                    verbose=False,
                    
                    
                )
    
                    
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
                #mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
                mask = torch.clamp(batch["mask"], min=0.0, max=1.0)
                predicted_image = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                ) #MJ: predicted_image is the generated image for the mask region

                inpainted = (1 - mask) * image + mask * predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
                
                # inpainted = predicted_image
                # inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                # Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
                
                
