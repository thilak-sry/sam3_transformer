from transformers import Sam3Processor, Sam3Model, Sam3VideoModel, Sam3VideoProcessor
import torch
from PIL import Image
import requests
import numpy as np
import matplotlib
import av


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

image_model = None
image_processor = None
video_model = None
video_processor = None

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_model_and_processor():
    global image_model, image_processor
    if image_model is None:
        try:
            image_model = Sam3Model.from_pretrained("facebook/sam3", local_files_only=True).to(device)
            image_processor = Sam3Processor.from_pretrained("facebook/sam3", local_files_only=True)
            print("Loaded image model and processor from local cache.")
        except Exception as e:
            print(f"Warning: Local loading failed ({e}). Attempting to load online from HuggingFace...")
            image_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
            image_processor = Sam3Processor.from_pretrained("facebook/sam3")
    return image_model, image_processor

def get_video_model_and_processor():
    global video_model, video_processor
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if video_model is None:
        try:
            video_model = Sam3VideoModel.from_pretrained("facebook/sam3", local_files_only=True).to(device, dtype=dtype)
            video_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3", local_files_only=True)
            print("Loaded video model and processor from local cache.")
        except Exception as e:
            print(f"Warning: Local video loading failed ({e}). Attempting to load online from HuggingFace...")
            video_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
            video_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    return video_model, video_processor




def run_inference(image: Image.Image, text: str, threshold: float = 0.5, mask_threshold: float = 0.5):
    model, processor = get_image_model_and_processor()
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
        
        # Draw solid colored masks on a black background
        bg = Image.new("RGBA", image.size, color=(0, 0, 0, 255))
        masks_np = 255 * masks_tensor.cpu().numpy().astype(np.uint8)
        
        n_masks = masks_np.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

        for mask, color in zip(masks_np, colors):
            mask_pil = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (255,))
            overlay.putalpha(mask_pil)
            bg = Image.alpha_composite(bg, overlay)
            
        segmented_image = bg.convert("RGB")
    else:
        annotated_image = image.convert("RGB")
        segmented_image = Image.new("RGB", image.size, color="black")
        
    return annotated_image, segmented_image, num_objects

def write_video(frames, output_path, fps=30):
    container = av.open(output_path, mode='w')
    try:
        stream = container.add_stream('libx264', rate=fps)
    except Exception:
        stream = container.add_stream('mpeg4', rate=fps)
        
    width, height = frames[0].size
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    
    for frame in frames:
        av_frame = av.VideoFrame.from_image(frame)
        for packet in stream.encode(av_frame):
            container.mux(packet)
            
    for packet in stream.encode():
        container.mux(packet)
        
    container.close()

def run_video_inference(video_frames, prompt: str, threshold: float = 0.5):
    video_model, video_processor = get_video_model_and_processor()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Initialize video inference session
    inference_session = video_processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    
    # Add text prompt to detect and track objects
    inference_session = video_processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt,
    )
    
    # Process all frames in the video
    outputs_per_frame = {}
    for model_outputs in video_model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=len(video_frames)
    ):
        processed_outputs = video_processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        
    annotated_frames = []
    segmented_frames = []
    
    for i, frame in enumerate(video_frames):
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
            
        frame_outputs = outputs_per_frame.get(i, None)
        
        if frame_outputs is not None and len(frame_outputs.get("object_ids", [])) > 0:
            masks_tensor = frame_outputs["masks"]
            if isinstance(masks_tensor, list):
                masks_tensor = torch.stack([m.squeeze() for m in masks_tensor])
            else:
                masks_tensor = masks_tensor.squeeze()
                
            if masks_tensor.ndim == 2:
                masks_tensor = masks_tensor.unsqueeze(0)
                
            # Overlay masks on the original frame
            annotated_frame = overlay_masks(frame_pil, masks_tensor).convert("RGB")
            
            # Create solid colored masks on a black background
            bg = Image.new("RGBA", frame_pil.size, color=(0, 0, 0, 255))
            masks_np = 255 * masks_tensor.cpu().numpy().astype(np.uint8)
            
            n_masks = masks_np.shape[0]
            cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
            colors = [
                tuple(int(c * 255) for c in cmap(idx)[:3])
                for idx in range(n_masks)
            ]

            for mask, color in zip(masks_np, colors):
                mask_pil = Image.fromarray(mask)
                overlay = Image.new("RGBA", frame_pil.size, color + (255,))
                overlay.putalpha(mask_pil)
                bg = Image.alpha_composite(bg, overlay)
                
            segmented_frame = bg.convert("RGB")
        else:
            annotated_frame = frame_pil.convert("RGB")
            segmented_frame = Image.new("RGB", frame_pil.size, color="black")
            
        annotated_frames.append(annotated_frame)
        segmented_frames.append(segmented_frame)
        
    return annotated_frames, segmented_frames

if __name__ == "__main__":
    import os
    # Load image
    image_path = "./face.jpg" if os.path.exists("./face.jpg") else "./0.jpg"
    image = Image.open(image_path).convert("RGB")

    # Segment using text prompt
    annotated_image, segmented_image, num_objects = run_inference(image, "eyes")

    print(f"Found {num_objects} objects")

    if num_objects > 0:
        output_path = "./output.jpg"
        annotated_image.save(output_path)
        print(f"Saved annotated image to {output_path}")
        
        output_segmented_path = "./output_segmented.jpg"
        segmented_image.save(output_segmented_path)
        print(f"Saved segmented image to {output_segmented_path}")
    else:
        print("No objects found to draw.")

