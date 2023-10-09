import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

import sys
print('sys.path', sys.path)
sys.path.append('..')
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


import torch
import torchvision.transforms as transforms
from PIL import Image

def image_average(image_path):
    """
    Compute the deviations of each pixel from the average color of the image using GPU.

    :param image_path: Path to the image file.
    :return: Tensor containing deviations for each pixel.
    """
    
    # Check if GPU (CUDA) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        raise RuntimeError("CUDA is not available on this machine.")
    
    # Open the image
    img = Image.open(image_path)
    
    # Convert the image to RGB
    img = img.convert("RGB")
    
    # Use torchvision's transform to convert the PIL image to a PyTorch tensor
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension
    
    # Move tensor to GPU
    img_tensor = img_tensor.to(device)
    
    # Calculate the mean of each channel: R, G and B
    mean_values = img_tensor.mean([2, 3]).squeeze()
    
    # Compute the deviations
    deviations = img_tensor - mean_values
    
    # Remove batch dimension and return the deviations
    return deviations.squeeze()



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
            for image_path, mask_path in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image_path)[1])
                
                batch = make_batch(image_path, mask_path, device=device)

                 #MJ: 
                init_image=batch["image"]  # in [-1,1]
                
                org_mask= batch["mask"]  # in [-1,1]
                b,c,h,w=org_mask.shape
                
                #org_mask = (org_mask + 1.0)/2.0 #MJ: trial for debugging: org_mask in [0,1]
                c1 = torch.nn.functional.interpolate(org_mask, size=(h//4,w//4) ) 
                
                latent_mask = (c1 + 1.0)/2.0  # in [0,1]
                org_mask = (org_mask + 1.0)/2.0
                #MJ: size=(H,W) of the masked_image which is in the latent space
                # encode masked image and concat downsampled mask
                #MJ: The masked_image is put to the cond_stage_model?
                #It is because of     cond_stage_config: __is_first_stage__ in config
                encoder_posterior = model.cond_stage_model.encode(batch["masked_image"]) 
                #MJ: encode into the 1/4 space
                #=> self.first_stage_model.encode(x): class VQModelInterface(VQModel): cond_stage_model = first_stage_model
                #c2: [1,3,128,128]
                # def encode(self, x):
                #     h = self.encoder(x) #MJ: self.encoder = Encoder(**ddconfig); => self.encoder.forward(x)
                #     #MJ: 
                #     h = self.quant_conv(h)
                #     return h
                c2 = model.get_first_stage_encoding(encoder_posterior) 
                #: => c2 = encoder_posterior / std(z) = encoder_posterior * vae_scale_factor      
                #: This is scale_factor which is used to scale the latents produced by the autoencoder before they are fed to the unet.
                #https://github.com/huggingface/diffusers/issues/437:
                #=> From Robin Rombach: 
                
                #1) We introduced the scale factor in the latent diffusion paper.
                # The goal was to handle different latent spaces (from different autoencoders, 
                # which can be scaled quite differently than images) with similar noise schedules. The scale_factor ensures that the initial latent space 
                # on which the diffusion model is operating has approximately unit variance. Hope this helps :)
               
                #2) To make sure I'm understanding, it sounds like you arrived at scale_factor = 0.18215 by averaging over a bunch of examples generated by the vae,
                # in order to ensure they have unit variance with the variance taken over all dimensions simultaneously? And scale_factor = 1 / std(z), schematically?
                
                #c_cat = torch.cat((c1, c2), dim=1) #MJ: c= the stack of the masked_image and the mask
                c_cat = torch.cat((c2, c1), dim=1) #MJ: c= the stack of the masked_image and the mask
                shape = (c_cat.shape[1] - 1,) + c_cat.shape[2:] #MJ: c=(B,3+1,H,W): shape=(3,H,W)= the shape of the image
                #MJ: I modified omri's sampler.sample() call, by providing mask and init_image as additional parameters
                samples_ddim, intermediates = sampler.sample(
                    S=opt.steps,
                    
                    conditioning=c_cat,
                    
                    batch_size=c_cat.shape[0],
                    shape=shape,
                    #mask = None,
                    mask=latent_mask,   #MJ: latent_mask in [0,1]
                                    
                    init_image = init_image,  #MJ: in [-1,1]                  
                                   
                    verbose=False,
                    
                    
                )
                                 
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                #MJ: batch is on the gpu
                image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
                predicted_image = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                ) #MJ: predicted_image is the generated image for the mask region

                #MJ: compute the average of the original image, image.
                # [2,3] refers to (H,W) dim of of the image tensor (B,C,H,W)
                mean_org  = image.mean([2,3], keepdim=True)
                mean_pred = predicted_image.mean([2,3], keepdim=True)
                #deviations_predicted = predicted_image - mean_pred
                # # Use mean_of_org_img to shift the mean color of the predicted image to the original image
                #predicted_image = mean_org + deviations_predicted
                
                std_org = image.std([2,3], keepdim=True)
                std_pred = predicted_image.std([2,3], keepdim=True)
                
                normalized_predicted = predicted_image - mean_pred
                
                #MJ: scale the stds of both images:
                # If the standard deviation of the predicted image (std_pred) is higher than that of the original image 
                #  (std_org), this operation will reduce the spread of pixel values in the predicted_image, and vice versa.
                predicted_image =   normalized_predicted * (std_org / std_pred) + mean_org
                #
                 
                predicted_image = torch.clamp(
                       predicted_image, min=0.0, max=1.0
                ) 
                 
                inpainted = (1 - mask) * image + mask * predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
