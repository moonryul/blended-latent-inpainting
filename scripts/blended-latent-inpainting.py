import argparse, os, sys, glob

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

#from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline ## This inpaint pipleline is  not pure, having text prompt


import torch


class BlendedLatentDiffusion:
    
    def __init__(self):
        self.parse_args()
        
        self.load_models()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument(
        "--prompt", type=str, #required=True,
                    default="a beatiful cat",help="The target text prompt"
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
            #default="stabilityai/stable-diffusion-2-1-base",
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

        self.args = parser.parse_args()
    #End def parse_args(self)    

    def load_models(self):
        
        
        #MJ: create a basic txt2img pipeline => Replace it with the pure inpainting pipeline
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     self.args.model_path, torch_dtype=torch.float16
        # )
        
        
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16
        )
        self.vae = pipe.vae.to(self.args.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.args.device)
        self.unet = pipe.unet.to(self.args.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    #End def load_models(self)
    
    @torch.no_grad()
    def blended_diffusion(
        self,
        image_path,
        mask_path,
        prompts_batched,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
        blending_percentage=0.25,
    ):
        batch_size = len(prompts_batched)
      
        image = Image.open(image_path)
        image = image.resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        
        #MJ: z_{init} ~ E(x), x= source image: z_{init} = source_latents
        
        source_latents = self._image2latent(image)
        #MJ: =>  latents = self.vae.encode(image)["latent_dist"].mean
              
        #MJ:  m_{latent} = downsample(m): resize the mask to dest_size=(64, 64): m_{latent} = latent_mask
        latent_mask, org_mask = self._read_mask(mask_path)  #MJ: resize mask to (64,64)
        
        masked_image_latents = source_latents * (latent_mask  < 0.5) #MJ: get the background image
               
        text_input = self.tokenizer(
            prompts_batched,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]

        max_length = text_input.input_ids.shape[-1]
        
        uncond_input = self.tokenizer(
            [""] * batch_size,    #MJ: consider the batch_size
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) #MJ: text_embeddings: shape=(8,77,1024)

        #MJ: z_{k} ~ noise(z_init, k): z_{k} = latents 
        # latents = torch.randn(
        #     (batch_size, self.unet.in_channels, height // 8, width // 8),  #MJ: consider the batch_size
        #     generator=generator,
        # )
        
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),  #MJ: consider the batch_size
            generator=generator,
        )
        
        
        
        latents = latents.to("cuda").half()

        self.scheduler.set_timesteps(num_inference_steps)

        #MJ: blending_percentage = 0.25 = denoising_strength
        #MJ: The denoising loop: self.scheduler.timesteps contains real timesteps in the range of 1000
         
        latent_mask = torch.cat([latent_mask]*2)            
        masked_image_latents = torch.cat([ masked_image_latents]*2) 
          
        for t in self.scheduler.timesteps[
            int(len(self.scheduler.timesteps) * blending_percentage) :
        ]:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2) #MJ: latents: shape=(4,9,64,64); latent_model_input: shape = torch.Size([8, 9, 64, 64])
              
            #MJ: https://github.com/huggingface/diffusers/pull/637
            #The functions: scale_initial_noise and scale_model_input are not required by DDIM or PNDM but just K-LMS. 
            # 
            latent_model_input = self.scheduler.scale_model_input(
               latent_model_input, timestep=t
            )

            
            #MJ: Add this line to the original blended-latent-diffusion, which uses
            # StableDiffusionPipeline (txt2imge), which has num_channels_unet=self.unet.config.in_channels =9

            #latent_model_input = torch.cat([latent_model_input, latent_mask, masked_image_latents], dim=1) 
            #latent_model_input = torch.cat([latent_model_input,  masked_image_latents, latent_mask], dim=1) 
             #MJ: latent_mask ([1,1,64,64]) and masked_image_latents (1,4,64,64]) are broadcasted to latent_model_input ([8,9,64,64])??
          
            #MJ: z_fg ~ denoise(z_t, d, t), d= text_embeddings
            # predict the noise residual
            with torch.no_grad():
                 noise_pred = self.unet(
                     latent_model_input, t,  encoder_hidden_states=text_embeddings #MJ: text_embeddings: shape = (8,77,1024)
                 ).sample
            #MJ: noise_pred is the model_output
            
                 
            #MJ: In the  ddpm code: 
            #    noise_pred = self.ddpm.apply_model(
            #         latent_model_input, t, cond, # context
            #     )
                     
             # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) #MJ: noise_pred: (8,4,64,64); noise_pred_uncond: (4,4,64,64)
            
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )          

            #MJ:  z_fg = latents
            # compute the previous noisy sample x_t -> x_t-1:  def step(self, model_output: torch.FloatTensor,timestep: int,sample: torch.FloatTensor,
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample #MJ: prev_sample is the field variable of SchedulerOutput dataclass
            
            #MJ:  latent_model_input.shape: torch.Size([4, 4, 64, 64])
            #            noise_pred.shape:  torch.Size([4, 4, 64, 64])
            #                 latents.shape: torch.Size([4, 9, 64, 64])
            # Blending
            #MJ: z_bg ~ noise(z_init, t), z_init= source_latents: z_bg = noise_source_latents
            noise_source_latents = self.scheduler.add_noise(
                
                source_latents, torch.randn_like(latents), t
            )
            #MJ: scheduler.add_noise(): #MJ: x_t = sqrt( alpha_t^hat) x0 + sqrt( 1-alpha_t^hat) eps
            
            #MJ: masking for blended latent diffusion:
            # z_t = z_fg * m_{latent} + z_bg * (1- m_{latent}) 
            latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)
        #for t in self.scheduler.timesteps[
             
        latents = 1 / 0.18215 * latents 
        #MJ: = 1/scale_factor * latents = std(z) * latents => from  the normalized space with unit variance to real value space

        with torch.no_grad():
            image = self.vae.decode(latents).sample
            

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
        
        latents = self.vae.encode(image)["latent_dist"].mean
        #MJ: latents = self.ddpm.first_stage_mode.encode(image)
        latents = latents * 0.18215 
        #MJ: = latents * scale_factor = latents/std(z) = the normalized values with unit variance

        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        #MJ: preserve the mask, not invert it
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.args.device)

        return mask, org_mask
#End class BlendedLatnetDiffusion

#MJ: python scripts/text_editing_stable_diffusion.py --prompt "a stone" --init_image "inputs/img.png" --mask "inputs/mask.png"

if __name__ == "__main__":
    
    bld = BlendedLatentDiffusion()
    
   #MJ results = bld.edit_image(
    results = bld.blended_diffusion(
        bld.args.init_image,
        bld.args.mask,
        prompts_batched=[bld.args.prompt] * bld.args.batch_size, #  MJ: consider the batch_size
        blending_percentage=bld.args.blending_start_percentage,
    )
    results_flat = np.concatenate(results, axis=1)
    Image.fromarray(results_flat).save(bld.args.output_path)
