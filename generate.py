import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    EulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    DiffusionPipeline,  # For SDXL
    FluxPipeline  # For FLUX
)
import argparse
import numpy as np
import os
from PIL import Image
from pytorch_lightning import seed_everything
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Modify CLIP embeddings and generate images')
    parser.add_argument('--model_path', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--codebook_dir', type=str, default="codebook_copro_sexual2")
    parser.add_argument('--prompts_csv', type=str, default=None,
                       help='Path to CSV file containing prompts')
    parser.add_argument('--prompt', type=str, default="a photo of a dog",
                       help='Single prompt to use when prompts_csv is not provided')
    parser.add_argument('--negative_prompt', type=str, default=None,
                       help='Negative prompt to use (only for SD v1.x models)')
    parser.add_argument('--start_idx', type=int, default=1,
                       help='Starting index for prompts')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='Ending index for prompts (None for all)')
    # Image generation arguments
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--output_path', type=str, default="output.png")
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    # Custom CLIP encoder arguments
    parser.add_argument('--training_method', type=str, default=None,
                       choices=['des', 'advunlearn', 'visu', 
                                'uce', 'esd', 'fmn', 'salun', 'spm', 'safegen', None],
                       help='Training method used (if None, use original CLIP)')
    parser.add_argument('--text_encoder_path', type=str, default=None,
                       help='Path to trained checkpoint')
    parser.add_argument('--text_encoder_2_path', type=str, default=None,
                       help='Path to trained checkpoint')
    parser.add_argument('--text_encoder_3_path', type=str, default=None,
                       help='Path to trained checkpoint')
    # Model type argument
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['sd_v1', 'sd_v2', 'sd_v3', 'sdxl', 'flux'],
                       help='Explicitly specify the model type (optional)')
    # Add SDXL refiner arguments
    parser.add_argument('--use_refiner', action='store_true',
                       help='Use SDXL refiner (only applicable for SDXL models)')
    parser.add_argument('--refiner_path', type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0",
                       help='Path to SDXL refiner model')
    parser.add_argument('--high_noise_frac', type=float, default=0.8,
                       help='Fraction of noise steps to run on base model vs refiner (only for SDXL with refiner)')
    args = parser.parse_args()

    device = torch.device(args.device)
    seed_everything(args.seed)
    
    # Check which model type to use, either from explicit arg or by inferring from path
    model_type = args.model_type
    if model_type is None:
        if "stable-diffusion-3" in args.model_path:
            model_type = "sd_v3"
        elif "stable-diffusion-xl" in args.model_path or "sdxl" in args.model_path.lower():
            model_type = "sdxl"
        elif "flux" in args.model_path.lower():
            model_type = "flux"
        elif "stable-diffusion-2" in args.model_path:
            model_type = "sd_v2"
        else:
            model_type = "sd_v1"  # Default to v1.x
    
    print(f"Using model type: {model_type} from path: {args.model_path}")
    
    # Load pipeline based on model type
    if model_type == "sd_v3":
        print(f"Loading Stable Diffusion V3: {args.model_path}")
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model_type == "sdxl":
        print(f"Loading Stable Diffusion XL base: {args.model_path}")
        pipe = DiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        
        # Initialize refiner if requested
        refiner = None
        if args.use_refiner:
            print(f"Loading Stable Diffusion XL refiner: {args.refiner_path}")
            refiner = DiffusionPipeline.from_pretrained(
                args.refiner_path,
                text_encoder_2=pipe.text_encoder_2,  # Share text_encoder_2
                vae=pipe.vae,  # Share VAE
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to(device)
    elif model_type == "flux":
        print(f"Loading FLUX: {args.model_path}")
        pipe = FluxPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16
        ).to(device)
        # pipe.enable_model_cpu_offload()
    elif model_type == "sd_v2":
        print(f"Loading Stable Diffusion V2: {args.model_path}")
        # For SD v2, use Euler scheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path, 
            scheduler=scheduler,
            safety_checker=None
        ).to(device)
    else:  # sd_v1
        print(f"Loading Stable Diffusion V1.x: {args.model_path}")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            safety_checker=None
        ).to(device)
            
        # Set scheduler to DDIM for V1
        pipe.scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")

    # Load text encoder based on training method
    if args.training_method == 'des':
        print(f'Training method: {args.training_method}')
        if args.text_encoder_path:
            print('Load DES text encoder')
            checkpoint = torch.load(args.text_encoder_path, map_location=device)
            missing_keys, unexpected_keys = pipe.text_encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Warning: Missing keys in text_encoder: {missing_keys}")
            print(f"Warning: Unexpected keys in text_encoder: {unexpected_keys}")
            # Free up GPU memory after loading weights
            del checkpoint
            torch.cuda.empty_cache()
        if args.text_encoder_2_path:
            print('Load DES text encoder 2')

            # For FLUX model, load weights on CPU first then move back to GPU
            if model_type == "flux":
                print('GPU memory optimization: Loading text_encoder_2 weights on CPU for FLUX model')
                original_device = pipe.text_encoder_2.device
                pipe.text_encoder_2 = pipe.text_encoder_2.to('cpu')
                checkpoint = torch.load(args.text_encoder_2_path, map_location='cpu')
                missing_keys, unexpected_keys = pipe.text_encoder_2.load_state_dict(checkpoint['model_state_dict'], strict=False)
                pipe.text_encoder_2 = pipe.text_encoder_2.to(original_device)
                print(f"Text encoder 2 moved back to {original_device}")
            else:
                checkpoint = torch.load(args.text_encoder_2_path, map_location=device)
                missing_keys, unexpected_keys = pipe.text_encoder_2.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Warning: Missing keys in text_encoder_2: {missing_keys}")
            print(f"Warning: Unexpected keys in text_encoder_2: {unexpected_keys}")
            
            if args.use_refiner and refiner is not None:
                print('Ensuring refiner uses updated text_encoder_2')
                refiner.text_encoder_2 = pipe.text_encoder_2
            
            # Free up GPU memory after loading weights
            del checkpoint
            torch.cuda.empty_cache()
        if args.text_encoder_3_path:
            print('Load DES text encoder 3')
            checkpoint = torch.load(args.text_encoder_3_path, map_location=device)
            missing_keys, unexpected_keys = pipe.text_encoder_3.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Warning: Missing keys in text_encoder_3: {missing_keys}")
            print(f"Warning: Unexpected keys in text_encoder_3: {unexpected_keys}")
            # Free up GPU memory after loading weights
            del checkpoint
            torch.cuda.empty_cache()
    elif args.training_method == 'advunlearn':
        print(f'Training method: {args.training_method}')
        from transformers import CLIPTextModel
        pipe.text_encoder = CLIPTextModel.from_pretrained(
            "OPTML-Group/AdvUnlearn",
            subfolder="nudity_unlearned",
        ).to(device)
        # Clear cache after loading model
        torch.cuda.empty_cache()
    elif args.training_method == 'visu':
        print(f'Training method: {args.training_method}')
        from transformers import CLIPTextModel
        pipe.text_encoder = CLIPTextModel.from_pretrained(
            "aimagelab/safeclip_vit-l_14",
        ).to(device)
        # Clear cache after loading model
        torch.cuda.empty_cache()
    
    pipe.text_encoder.eval()

    # Handle prompts
    if args.prompts_csv:
        # Read CSV with header
        df = pd.read_csv(args.prompts_csv)
        prompts = df['prompt'].replace(r'^\s*$', np.nan, regex=True).dropna().tolist()

        # Adjust end index if not specified
        if args.end_idx is None:
            args.end_idx = len(prompts)
    else:
        # Use single prompt multiple times
        prompts = [args.prompt] * (args.end_idx - args.start_idx)

        # Don't use fixed seed when repeating the same prompt
        seed_everything(None)  # Use random seeds

    # Process each prompt in the specified range
    for i in range(args.start_idx, args.end_idx):
        if args.prompts_csv:
            prompt = prompts[i]
        else:
            prompt = args.prompt
            # Generate new random seed for each iteration
            current_seed = i
            seed_everything(current_seed)

        prompt_num = i

        print(f"\nProcessing prompt {prompt_num}/{args.end_idx}: {prompt}")

        # Generate image using pipeline with model-specific parameters
        generator = torch.Generator(device=device).manual_seed(current_seed) if 'current_seed' in locals() else None

        with torch.no_grad():
            # Model-specific generation parameters
            if model_type == "flux":
                image = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=512,  # FLUX-specific parameter
                    generator=generator
                ).images[0]
            elif model_type == "sdxl":
                # SDXL with optional refiner
                if args.use_refiner and refiner:
                    print(f"Using SDXL with refiner (high_noise_frac={args.high_noise_frac})")
                    # First pass with base model
                    latents = pipe(
                        prompt=prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        denoising_end=args.high_noise_frac,
                        output_type="latent",
                        generator=generator
                    ).images

                    # Second pass with refiner model
                    image = refiner(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        denoising_start=args.high_noise_frac,
                        image=latents,
                        generator=generator
                    ).images[0]
                else:
                    # Standard SDXL without refiner
                    image = pipe(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator
                    ).images[0]
            elif model_type == "sd_v3":
                image = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator
                ).images[0]
            else:
                # Check if negative prompt should be applied (only for v1.x)
                if model_type == "sd_v1" and args.negative_prompt is not None:
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator
                    ).images[0]
                else:
                    # Standard generation without negative prompt
                    image = pipe(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator
                    ).images[0]

        # Create output filename and save
        output_filename = f"{args.output_path}_{args.training_method}_ip"
        output_filename = output_filename.replace("_ip", f"_{prompt_num}p")
        
        # Create appropriate output directory based on model type and training method
        model_name = args.model_path.split('/')[-1] if '/' in args.model_path else args.model_path
        output_dir = f"results/{args.training_method}/{model_name}/{args.output_path}"
        
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, output_filename + ".png")
        image.save(full_path)
        print(f"Generated image saved to: {full_path}")

if __name__ == "__main__":
    main() 