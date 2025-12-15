from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from models.pipeline_base import BasePipeline
from models.referNet.ReferenceNet_attention import forward_second_half
from models.referNet.utils import get_attn_module
from utils import change_forward_fn, get_attn_module_for_correct_order


# Unidirectional interaction
def single_Roomeditor_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    save_features: bool = False,
):
    assert not self.use_ada_layer_norm, "use_ada_layer_norm not supported in hacked_basic_transformer_forward_write"
    assert not self.use_ada_layer_norm_zero, "use_ada_layer_norm_zero not supported in hacked_basic_transformer_forward_write"
    assert not self.only_cross_attention, "only_cross_attention not supported in hacked_basic_transformer_forward_write"
    cross_attention_kwargs = cross_attention_kwargs or {}
    token_cnt_half = hidden_states.shape[1] // 2
    norm_hidden_states = self.norm1(hidden_states)
    # 1. Self-Attention on reference tokens
    attn_output_ref = self.attn1(
        norm_hidden_states[:, token_cnt_half:, :],
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    # 1b. Self-Attention on target tokens
    attn_output_tar = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )[:, :token_cnt_half, :]
    if save_features:
        self.attn_output = attn_output_ref
        self.hidden_input = norm_hidden_states
    # Concatenate outputs back in token order
    attn_output = torch.cat([attn_output_tar, attn_output_ref], dim=1)
    hidden_states = attn_output + hidden_states

    # Second half of the Transformer block
    hidden_states = forward_second_half(
        hidden_states, encoder_hidden_states, encoder_attention_mask, cross_attention_kwargs,
        self.attn2, self.norm2, self.norm3, self.ff
    )
    return hidden_states

# No interaction between reference and target
def No_Roomeditor_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
):
    assert not self.use_ada_layer_norm, "use_ada_layer_norm not supported in hacked_basic_transformer_forward_write"
    assert not self.use_ada_layer_norm_zero, "use_ada_layer_norm_zero not supported in hacked_basic_transformer_forward_write"
    assert not self.only_cross_attention, "only_cross_attention not supported in hacked_basic_transformer_forward_write"
    cross_attention_kwargs = cross_attention_kwargs or {}
    token_cnt_half = hidden_states.shape[1] // 2
    norm_hidden_states = self.norm1(hidden_states)
    # Self-Attention separately on reference and target
    attn_output_ref = self.attn1(
        norm_hidden_states[:, token_cnt_half:, :],
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    attn_output_tar = self.attn1(
        norm_hidden_states[:, :token_cnt_half, :],
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    attn_output = torch.cat([attn_output_tar, attn_output_ref], dim=1)
    hidden_states = attn_output + hidden_states

    # Second half of the Transformer block
    hidden_states = forward_second_half(
        hidden_states, encoder_hidden_states, encoder_attention_mask, cross_attention_kwargs,
        self.attn2, self.norm2, self.norm3, self.ff
    )
    return hidden_states


def hack_attn_forward(unet):
    unet = unet.module if hasattr(unet, "module") else unet
    down_attn_modules = get_attn_module(unet, mode="down")
    midup_attn_modules = get_attn_module(unet, mode="midup")

    # Disable interaction in down; mid and up unidirectional
    for module in down_attn_modules:
        change_forward_fn(module, No_Roomeditor_forward)
    for module in midup_attn_modules:
        change_forward_fn(module, single_Roomeditor_forward)
    return

class Roomeditor_Pipeline(BasePipeline):
    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        source_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        concat_dim: int = -2,
        save_attn_score=False,
    ):
        # Default height and width to UNet configuration
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        batch_size = len(image) if isinstance(image, list) else image.shape[0]
        device = self._execution_device

        # Set timesteps for scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        if num_inference_steps < 1:
            raise ValueError(
                f"Adjusted number of pipeline steps ({num_inference_steps}) < 1"
            )

        # Handle classifier-free guidance embeddings
        if self.do_classifier_free_guidance and negative_prompt_embeds is not None and prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Prepare background conditioned image
        init_image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        # Prepare reference image
        source_init_image = self.image_processor.preprocess(source_image, height=height, width=width).to(dtype=torch.float32)

        num_channels_latents = self.vae.config.latent_channels
        # Prepare initial noisy latents based on concatenation dimension
        if concat_dim == -2:
            new_height = height * 2
            new_width = width
        else:
            new_height = height
            new_width = width * 2
        latents = self.prepare_noise(
            batch_size,
            num_channels_latents,
            new_height,
            new_width,
            self.unet.dtype,
            device,
            generator,
        )
        # Split latents to get noise for reference
        _, noise = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)

        # Encode reference image latents
        source_image_latents = self.prepare_image_latents(
            self.unet.dtype, device, generator, source_init_image
        )
        # Prepare mask condition latents
        mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width)
        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            init_image,
            height,
            width,
            self.unet.dtype,
            device,
            generator,
            do_classifier_free_guidance=False
        )
        if self.do_classifier_free_guidance:
            mask = torch.cat([mask] * 2)
        mask_latents_concat = torch.cat([mask, torch.zeros_like(mask)], dim=concat_dim)
        masked_image_latents_concat = torch.cat([masked_image_latents, source_image_latents], dim=concat_dim)

        if self.do_classifier_free_guidance:
            uc_latents = torch.cat([masked_image_latents, torch.zeros_like(source_image_latents)], dim=concat_dim)
            masked_image_latents_concat = torch.cat([uc_latents, masked_image_latents_concat])

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        self._num_timesteps = len(timesteps)
        attn_score_list = {}
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                back = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
                latents = torch.cat([back, source_image_latents], dim=concat_dim)

                # Prepare model input for guidance
                if self.do_classifier_free_guidance:
                    back = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
                    uc_latents = torch.cat([back, torch.zeros_like(source_image_latents)], dim=concat_dim)
                    latent_model_input = torch.cat([uc_latents, latents])
                else:
                    latent_model_input = latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat(
                    [latent_model_input, mask_latents_concat, masked_image_latents_concat], dim=1
                )

                # Predict noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                if save_attn_score:
                    attn_score = get_attn_scores(self.unet, batch_size, i)
                    if attn_score is not None:
                        attn_score_list.update(attn_score)

                # Apply classifier-free guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                progress_bar.update()

        # Decode final image
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        image, has_nsfw_concept = self.decode_image(latents, device, generator, output_type)

        # Free CPU offloaded models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        if save_attn_score:
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), attn_score_list
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


# Helper to split attention scores by batch
def process_attn_score(score):
    # Score shape: (8 * batch, 2T, 2T) where 2T due to concatenated pair and 8 attention heads
    score = score.cpu()
    _, score = score.chunk(2)
    T = score.shape[1] // 2
    score = score.reshape(-1, 8, 2*T, 2*T)
    # Average over heads
    score = score.mean(dim=1)
    # Split per sample
    return [s.squeeze(0) for s in score.chunk(score.shape[0], dim=0)]

# Gather attention scores from selected blocks every 3rd module
def get_attn_scores_for_model(unet, mode, score_list):
    modules = get_attn_module_for_correct_order(unet, mode=mode)
    for i, module in enumerate(modules):
        if i % 3 == 0:
            scores = process_attn_score(module.attn_score)
            for j, s in enumerate(scores):
                score_list[j]["score"][f"{mode}_{i}"] = s
    return score_list

# Full attention score collection at specific steps
def get_attn_scores(unet, batch_size, step):
    if step not in [0, 9, 19, 29, 39, 49]:
        return None
    print(f"Getting attn scores for step {step}...")
    score_list = [{"score": {}} for _ in range(batch_size)]
    score_list = get_attn_scores_for_model(unet, 'down', score_list)
    score_list = get_attn_scores_for_model(unet, 'mid', score_list)
    score_list = get_attn_scores_for_model(unet, 'up', score_list)
    return {step: score_list}