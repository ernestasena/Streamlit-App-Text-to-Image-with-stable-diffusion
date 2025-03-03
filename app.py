from diffusers import StableDiffusionPipeline # type: ignore
import torch # type: ignore

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a plar bear on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")

