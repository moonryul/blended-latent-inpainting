"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,            #MJ: provide mask (latent_mask)
        org_mask=None,        #MJ: provide mask in the original image
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None, #MJ: Aha! When we use the stable diffusion for inpainting, we have an option to set unconditional_conditioning to None
        skip_steps=0,
        init_image=None,  #MJ: the original input image
        percentage_of_pixel_blending=0, #MJ: provide percentage_of_pixel_blending = 0.1
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None: #MJ: check the condition
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for DDIM sampling is {size}, eta {eta}")

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            org_mask=org_mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            skip_steps=skip_steps,
            init_image=init_image,
            percentage_of_pixel_blending=percentage_of_pixel_blending,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        org_mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        skip_steps=0,
        init_image=None,
        percentage_of_pixel_blending=0,
    ):
        device = self.model.betas.device
        b = shape[0]

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0])
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        if skip_steps != 0:
            timesteps = timesteps[:-skip_steps]

        time_range = (
            reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps) #MJ: timesteps: (50,)
        ) #MJ: time_range: [981,961,941,....,21,1]: step=20
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        #MJ: compute x0 and x_T from init_image
        if init_image is not None:
            assert (
                x0 is None and x_T is None
            ), "Try to infer x0 and x_t from init_image, but they already provided"

            encoder_posterior = self.model.encode_first_stage(init_image)
             #MJ: => self.first_stage_model.encode(x); encoder_posterior: [1,3,128,128]=mean ?
            x0 = self.model.get_first_stage_encoding(encoder_posterior) #MJ: x0 = the normalized latent image with unit variance
            last_ts = torch.full((1,), time_range[0], device=device, dtype=torch.long)
            x_T = torch.cat([self.model.q_sample(x0, last_ts) for _ in range(b)])
            img = x_T
            
        elif x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {"x_inter": [img], "pred_x0": [img]}

        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        cutoff_point = int(len(time_range) * (1 - percentage_of_pixel_blending)) #MJ:  percentage_of_pixel_blending=0; cutoff_point=len(rime_range) 
        
        #MJ: The denoising loop: img = x_T, x0 is computed
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

                
            #MJ:1)  noise_pred = self.unet(
                #     latent_model_input, t, encoder_hidden_states=text_embeddings
                # ).sample:
                #2)   #MJ:  z_fg = latents
            #    compute the previous noisy sample x_t -> x_t-1:  def step(self, model_output: torch.FloatTensor,timestep: int,sample: torch.FloatTensor,
            #    latents = self.scheduler.step(noise_pred, t, latents).prev_sample #MJ: prev_sample is the field variable of SchedulerOutput dataclass
            
            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            img, pred_x0 = outs

            #MJ: i = 0,1,.....,49
            # Latent-level blending: mask is the latent_mask
            if mask is not None and i < cutoff_point: #MJ: perform latent-level blending all the time in this experiment
                #n_masks = mask.shape[0]  #MJ: mask: shape=(1,1,128,128); 
                # masks_interval = len(time_range) // n_masks + 1
                # curr_mask = mask[i // masks_interval].unsqueeze(0) #MJ:mask[i // masks_interval]: shape=()1,128,128); curr_mask: shape =(1,1,128,128)
                # # print(f"Using index {i // masks_interval}")
                
                #MJ: get the blurred version of the original image x0 corresponding to time ts:
                img_orig = self.model.q_sample(x0, ts)
                #img = img_orig * (1 - curr_mask) + curr_mask * img
                img = img_orig * (1 - mask) + mask * img
            # Pixel-level blending: after cutoff_point: org_mask is the mask in the original image
            # if org_mask is not None and i >= cutoff_point:
            #     foreground_pixels = self.model.decode_first_stage(pred_x0)
            #     background_pixels = init_image
            #     pixel_blended = foreground_pixels * org_mask + background_pixels * (1 - org_mask)
                
            #     img_x0 = self.model.get_first_stage_encoding(
            #         self.model.encode_first_stage(pixel_blended)
            #     )
            #     img = self.model.q_sample(img_x0, ts)

            #MJ: img may come from img,pred_x0 = outs or img =  self.model.q_sample(img_x0, ts)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)
        #MJ:  for i, step in enumerate(iterator)
        
        return img, intermediates
    #End  def ddim_sampling
    
    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        b, *_, device = *x.shape, x.device

        #MJ: compute the model output e_t:
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(x, t, c)
            
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        #MJ: get the parameters needed to compute the current prediction for x_0, pred_x0,
        #    and the previous sample, x_prev
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep, index:
        a_t = torch.full( (b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            
        #MJ: compute the previous sample, x_prev    
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
