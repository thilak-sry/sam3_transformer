from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests
import numpy as np
import matplotlib

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = Sam3Model.from_pretrained("facebook/sam3", local_files_only=True).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3", local_files_only=True)
    print("Loaded model and processor from local cache.")
except Exception as e:
    print(f"Warning: Local loading failed ({e}). Attempting to load online from HuggingFace...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")



def run_inference(image: Image.Image, text: str, threshold: float = 0.5, mask_threshold: float = 0.5):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    num_objects = len(results["masks"])

    if num_objects > 0:
        if isinstance(results["masks"], list):
            masks_tensor = torch.stack([m.squeeze() for m in results["masks"]])
        else:
            masks_tensor = results["masks"].squeeze()
            
        if masks_tensor.ndim == 2:
            masks_tensor = masks_tensor.unsqueeze(0)
            
        annotated_image = overlay_masks(image, masks_tensor).convert("RGB")
    else:
        annotated_image = image.convert("RGB")
        
    return annotated_image, num_objects

if __name__ == "__main__":
    # Load image
    image_path = "./0.jpg"
    image = Image.open(image_path).convert("RGB")

    # Segment using text prompt
    annotated_image, num_objects = run_inference(image, "eyes")

    print(f"Found {num_objects} objects")

    if num_objects > 0:
        output_path = "./output.jpg"
        annotated_image.save(output_path)
        print(f"Saved annotated image to {output_path}")
    else:
        print("No objects found to draw.")

