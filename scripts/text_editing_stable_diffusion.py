import argparse
import numpy as np
from PIL import Image

from diffusers import DDIMScheduler, Stablcd eDiffusionPipeline
import torch


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

    def load_models(self):
        
        #MJ: create a basic txt2img pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
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

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
        blending_percentage=0.25,
    ):
        batch_size = len(prompts)

        image = Image.open(image_path)
        image = image.resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        
        #MJ: z_{init} ~ E(x), x= source image: z_{init} = source_latents
        source_latents = self._image2latent(image)
              
        #MJ:  m_{latent} = downsample(m): resize the mask to dest_size=(64, 64): m_{latent} = latent_mask
        latent_mask, org_mask = self._read_mask(mask_path)
        

        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]

        max_length = text_input.input_ids.shape[-1]
        
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        #MJ: z_{k} ~ noise(z_init, k): z_{k} = latents 
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),  #MJ: self.unet.in_channels=4; => use 'unet.config.in_channels' 
            generator=generator,
        )
        
        latents = latents.to("cuda").half()

        self.scheduler.set_timesteps(num_inference_steps)

        #MJ: blending_percentage = 0.25
        for t in self.scheduler.timesteps[
            int(len(self.scheduler.timesteps) * blending_percentage) :
        ]:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            #MJ: z_fg ~ denoise(z_t, d, t), d= text_embeddings
            # predict the noise residual
            #MJ: use "stabilityai/stable-diffusion-2-1-base":
            
#             This stable-diffusion-2-1-base model fine-tunes stable-diffusion-2-base (512-base-ema.ckpt) with 220k extra steps taken, with punsafe=0.98 on the same dataset.

# Use it with the stablediffusion repository: download the v2-1_512-ema-pruned.ckpt here.
# Use it with 🧨 diffusers

#===> Now use: https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
# This stable-diffusion-2-inpainting model is resumed from stable-diffusion-2-base (512-base-ema.ckpt) and trained for another 200k steps. Follows the mask-generation strategy presented in LAMA which, in combination with the latent VAE representations of the masked image, are used as an additional conditioning.

# Use it with the stablediffusion repository: download the 512-inpainting-ema.ckpt here.
# Use it with 🧨 diffusers:

# pip install diffusers transformers accelerate scipy safetensors

# from diffusers import StableDiffusionInpaintPipeline
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
# #image and mask_image should be PIL images.
# #The mask structure is white for inpainting and black for keeping as is
# image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
# image.save("./yellow_cat_on_park_bench.png")

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

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

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
        image = image.half()
        
        latents = self.vae.encode(image)["latent_dist"].mean
        
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
    results = bld.edit_image(
        bld.args.init_image,
        bld.args.mask,
        prompts=[bld.args.prompt] * bld.args.batch_size,
        blending_percentage=bld.args.blending_start_percentage,
    )
    results_flat = np.concatenate(results, axis=1)
    Image.fromarray(results_flat).save(bld.args.output_path)
