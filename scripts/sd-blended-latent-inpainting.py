#MJ: import argparse
#import numpy as np
#from PIL import Image

import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

from diffusers import DDIMScheduler, StableDiffusionPipeline
##from diffusers import DDIMScheduler, StableDiffusionInPaintPipeline ## This inpaint pipleline is  not pure, having text prompt
#from diffusers import DDIMScheduler

import torch


def make_batch_sd(
        image,
        mask,
        #txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5) #MJ:  masked_image = (1 - mask) * image #MJ: the mask region becomes zero

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        #MJ: "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


class BlendedLatnetDiffusion:
    
    def __init__(self):
        self.parse_args()
        
        self.load_models()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt", type=str, required=True, help="The target text prompt"
        )
        parser.add_argument(
            "--init_image", type=str, required=True, help="The path to the input image"
        )
        parser.add_argument(
            "--mask", type=str, required=True, help="The path to the input mask"
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default="stabilityai/stable-diffusion-2-1-base",
            help="The path to the HuggingFace model",
        )
        parser.add_argument(
            "--batch_size", type=int, default=4, help="The number of images to generate"
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

        self.args = parser.parse_args()
    #End def parse_args(self)    

    def load_models(self):
        
        #MJ: create a basic txt2img pipeline
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     self.args.model_path, torch_dtype=torch.float16
        # )
        
        config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
        ddpm = instantiate_from_config(config.model)
        ddpm.load_state_dict(
               torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ddpm = ddpm.to(device)
        
        self.ddpm = ddpm
        self.sampler = DDIMSampler(self.ddpm)
        
        #MJ: self.vae = pipe.vae.to(self.args.device)
        
        
        # def encode_first_stage(self, x):
        # return self.first_stage_model.encode(x)
    
    
        #self.vae = ddpm.first_stage_model  #MJ: In our config, cond_stage_model = first_stage_model
        #MJ: self.tokenizer = pipe.tokenizer
        #MJ: self.text_encoder = pipe.text_encoder.to(self.args.device)
        #self.unet = pipe.unet.to(self.args.device)
        
        #self.unet = ddpm.model.to(self.args.device)
        #LatentDiffusion.apply_model(self, x_noisy, t, cond, return_ids=False):
        
        #MJ: self.scheduler = DDIMScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     clip_sample=False,
        #     set_alpha_to_one=False,
        # )
    #End def load_models(self)
    
    @torch.no_grad()
    def blended_inpaint(
        self,
        image_path,
        mask_path,
       # prompts, # MJ:
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
        blending_percentage=0.25,
    ):
        #MJ: batch_size = len(prompts)
      
        with torch.no_grad(), \
            torch.autocast("cuda"):
            
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
            num_samples = 1  #MJ:
            image = Image.open(image_path)
            image = image.resize((height, width), Image.BILINEAR)
            image = np.array(image)[:, :, :3]
            
            
            batch = make_batch_sd(image, mask, #MJ: txt=prompt,
                                device=device, num_samples=num_samples)
            
            
            #MJ: z_{init} ~ E(x), x= source image: z_{init} = source_latents
            
            source_latents = self._image2latent(image)
            #MJ: =>  latents = self.vae.encode(image)["latent_dist"].mean
                
            #MJ:  m_{latent} = downsample(m): resize the mask to dest_size=(64, 64): m_{latent} = latent_mask
            latent_mask, org_mask = self._read_mask(mask_path)
    
            
            
            c_cat = list()
            h,w = image.size
            
            for ck in self.ddpm.model.concat_keys: #MJ: "mask" or "masked_image"
                cc = batch[ck].float()
                if ck != self.ddpm.model.masked_image_key: #MJ: "mask"
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else: #MJ: masked_image
                    cc = self.ddpm.model.get_first_stage_encoding(
                        self.ddpm.model.encode_first_stage(cc))
                c_cat.append(cc)
                
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            #MJ: If conditionning_key = hybrid: cond: a dict => cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # Otherwise, cond is a list of images
            
            #MJ:
            # text_input = self.tokenizer(
            #     prompts,
            #     padding="max_length",
            #     max_length=self.tokenizer.model_max_length,
            #     truncation=True,
            #     return_tensors="pt",
            # )
            
            # text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]

            # max_length = text_input.input_ids.shape[-1]
            
            # uncond_input = self.tokenizer(
            #     [""] * batch_size,
            #     padding="max_length",
            #     max_length=max_length,
            #     return_tensors="pt",
            # )
            # uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]
            # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            #MJ: z_{k} ~ noise(z_init, k): z_{k} = latents 
            # latents = torch.randn(
            #     (batch_size, self.unet.in_channels, height // 8, width // 8),
            #     generator=generator,
            # )
            
            latents = torch.randn(
                (num_samples, self.ddpm.in_channels, height // 8, width // 8),
                generator=generator,
            )
            
            
            latents = latents.to("cuda").half()

            self.scheduler.set_timesteps(num_inference_steps)

            denoising_strength  = 0.25
            timesteps =  self.sampler.ddim_timesteps 
             
            for t in self.scheduler.timesteps[
                int(len(self.scheduler.timesteps) * denoising_strength) :
            ]:
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                #MJ: latent_model_input = torch.cat([latents] * 2)
                
                latent_model_input = torch.cat([latents] * 1)
                
                #MJ: https://github.com/huggingface/diffusers/pull/637
                #The functions: scale_initial_noise and scale_model_input are not required by DDIM or PNDM but just K-LMS. 
                # 
                #latent_model_input = self.scheduler.scale_model_input(
                #    latent_model_input, timestep=t
                #)

                #MJ: z_fg ~ denoise(z_t, d, t), d= text_embeddings
                # predict the noise residual
                # with torch.no_grad():
                #     noise_pred = self.unet(
                #         latent_model_input, t, # encoder_hidden_states=text_embeddings
                #     ).sample
            
                #MJ: construct cond
                with torch.no_grad():
                    noise_pred = self.ddpm.apply_model(
                        latent_model_input, t, c_cat # encoder_hidden_states=text_embeddings
                    )            

                #MJ: def apply_model(self, x_noisy, t, cond
                # perform guidance
                #MJ: noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                
                # noise_pred = noise_pred_uncond + guidance_scale * (
                #     noise_pred_text - noise_pred_uncond
                # )

                #MJ: z_fg ~ denoise(z_t,d,t): z_fg = latents
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # Blending
                #MJ: z_bg ~ noise(z_init, t), z_init= source_latents: z_bg = noise_source_latents
                noise_source_latents = self.scheduler.add_noise(
                    
                    source_latents, torch.randn_like(latents), t
                )
                #MJ: z_t = z_fg * m_{latent} + z_bg * (1- m_{latent}) 
                latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)
            #for t in self.scheduler.timesteps[
                
            latents = 1 / 0.18215 * latents 
            #MJ: = 1/scale_factor * latents = std(z) * latents => from  the normalized space with unit variance to real value space

            #MJ: with torch.no_grad():
            #     image = self.vae.decode(latents).sample
                
            x_samples_ddim = model.decode_first_stage(samples_cfg)    
                

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")

            return images
    #def blended_inpaint
    
    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
        image = image.half()
        
        #MJ: latents = self.vae.encode(image)["latent_dist"].mean
        latents = self.ddpm.first_stage_mode.encode(image)
        latents = latents * 0.18215 
        #MJ: = latents * scale_factor = latents/std(z) = the normalized values with unit variance

        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.args.device)

        return mask, org_mask

#MJ: python scripts/text_editing_stable_diffusion.py --prompt "a stone" --init_image "inputs/img.png" --mask "inputs/mask.png"

if __name__ == "__main__":
    
    bld = BlendedLatnetDiffusion()
    
   #MJ results = bld.edit_image(
    results = bld.blended_inpaint(
        bld.args.init_image,
        bld.args.mask,
       #MJ prompts=[bld.args.prompt] * bld.args.batch_size,
        blending_percentage=bld.args.blending_start_percentage,
    )
    results_flat = np.concatenate(results, axis=1)
    Image.fromarray(results_flat).save(bld.args.output_path)
