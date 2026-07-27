import base64
import io
import requests
from PIL import Image
import runpod
from inference import run_inference, run_video_inference, write_video
import os
import uuid

# Load environment variables from .env.local if present (for local testing)
env_path = os.path.join(os.path.dirname(__file__), '.env.local')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            trimmed = line.strip()
            if trimmed and not trimmed.startswith('#'):
                if '=' in trimmed:
                    key, value = trimmed.split('=', 1)
                    val = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = val

try:
    from azure.storage.blob import ContainerClient, ContentSettings
except ImportError:
    ContainerClient, ContentSettings = None, None

AZURE_STORAGE_SAS_URL = os.environ.get("AZURE_STORAGE_SAS_URL")
AZURE_READ_SAS_TOKEN = os.environ.get("AZURE_READ_SAS_TOKEN")

def upload_file_to_blob(local_path: str, content_type: str = None) -> str:
    if not AZURE_STORAGE_SAS_URL:
        return local_path
    if ContainerClient is None:
        print("Error: azure-storage-blob library is not installed.")
        return local_path
    try:
        container_client = ContainerClient.from_container_url(AZURE_STORAGE_SAS_URL)
        file_ext = os.path.splitext(local_path)[1]
        unique_name = f"{uuid.uuid4()}{file_ext}"
        blob_client = container_client.get_blob_client(unique_name)
        extra_kwargs = {}
        if content_type:
            extra_kwargs["content_settings"] = ContentSettings(content_type=content_type)
        print(f"Uploading {local_path} to blob {unique_name}...")
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, **extra_kwargs)
            
        url = blob_client.url.split('?', 1)[0]
        
        # Determine SAS token to append to returned URL
        sas_token = None
        if AZURE_READ_SAS_TOKEN:
            sas_token = AZURE_READ_SAS_TOKEN.lstrip('?')
        elif "?" in AZURE_STORAGE_SAS_URL:
            sas_token = AZURE_STORAGE_SAS_URL.split("?", 1)[1]
            
        if sas_token:
            url = f"{url}?{sas_token}"
            
        return url
    except Exception as e:
        print(f"Error uploading to Azure Blob: {e}")
        return local_path

def handler(event):
    print("Worker Start")
    input_data = event['input']
    
    image_input = input_data.get('image')
    video_input = input_data.get('video')
    prompt = input_data.get('prompt', 'eyes')  

    # Auto-detect if image input is actually a video
    if image_input and not video_input:
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif')
        clean_path = image_input.split('?', 1)[0]
        is_video_path = len(image_input) < 300 and any(clean_path.lower().endswith(ext) for ext in video_extensions)
        is_video_uri = image_input.startswith("data:video/")
        if is_video_path or is_video_uri:
            video_input = image_input
            image_input = None

    print(f"Received prompt: {prompt}")
    
    if video_input:
        import os
        from transformers.video_utils import load_video
        
        # Determine source type and save to temporary file if base64 or read URL
        if video_input.startswith("http"):
            video_path = video_input
        elif len(video_input) > 200:
            if "," in video_input:
                video_input = video_input.split(",", 1)[1]
            video_data = base64.b64decode(video_input)
            video_path = "./temp_input.mp4"
            with open(video_path, "wb") as f:
                f.write(video_data)
        else:
            video_path = video_input
            
        print(f"Loading video from {video_path}...")
        video_frames, _ = load_video(video_path)
        print(f"Loaded {len(video_frames)} frames.")
        
        # Perform SAM3 Video inference
        annotated_frames, segmented_frames = run_video_inference(video_frames, prompt)
        
        # Save output videos locally
        output_path = "./output.mp4"
        output_segmented_path = "./output_segmented.mp4"
        
        write_video(annotated_frames, output_path)
        print(f"Saved output video locally to {output_path}")
        
        write_video(segmented_frames, output_segmented_path)
        print(f"Saved segmented video locally to {output_segmented_path}")
        
        # Clean up temporary input file if created
        if not video_input.startswith("http") and len(video_input) > 200:
            try:
                os.remove(video_path)
            except Exception:
                pass
                
        # Read output videos and convert to base64 or upload to blob
        
        if AZURE_STORAGE_SAS_URL:
            annotated_url = upload_file_to_blob(output_path, content_type="video/mp4")
            segmented_url = upload_file_to_blob(output_segmented_path, content_type="video/mp4")
            
            # Clean up local output files
            try:
                os.remove(output_path)
                os.remove(output_segmented_path)
            except Exception:
                pass
                
            print("Successfully processed and uploaded video.")
            return {
                "video": annotated_url,
                "segmented_video": segmented_url
            }
        else:
            with open(output_path, "rb") as f:
                annotated_video_data = f.read()
            annotated_video_b64 = base64.b64encode(annotated_video_data).decode("utf-8")
            
            with open(output_segmented_path, "rb") as f:
                segmented_video_data = f.read()
            segmented_video_b64 = base64.b64encode(segmented_video_data).decode("utf-8")
            
            print("Successfully processed video locally.")
            return {
                "video": f"data:video/mp4;base64,{annotated_video_b64}",
                "segmented_video": f"data:video/mp4;base64,{segmented_video_b64}"
            }
    else:
        # Image path (default if no inputs specified)
        if not image_input:
            image_input = './face.jpg'
            
        # Load image (supports URL, base64, or local path)
        if image_input.startswith("http"):
            response = requests.get(image_input)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        elif len(image_input) > 200:
            if "," in image_input:
                image_input = image_input.split(",", 1)[1]
            image_data = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
        
        # Perform SAM3 inference
        annotated_image, segmented_image, num_objects = run_inference(image, prompt)
        
        # Save the output image locally
        output_path = "./output.jpg"
        annotated_image.save(output_path)
        print(f"Saved output image locally to {output_path}")
        
        output_segmented_path = "./output_segmented.jpg"
        segmented_image.save(output_segmented_path)
        print(f"Saved segmented image locally to {output_segmented_path}")
        
        # Convert output image to base64 or upload to blob
        
        if AZURE_STORAGE_SAS_URL:
            annotated_url = upload_file_to_blob(output_path, content_type="image/jpeg")
            segmented_url = upload_file_to_blob(output_segmented_path, content_type="image/jpeg")
            
            # Clean up local output files
            try:
                os.remove(output_path)
                os.remove(output_segmented_path)
            except Exception:
                pass
                
            print(f"Successfully processed and uploaded image. Found {num_objects} objects.")
            return {
                "image": annotated_url,
                "segmented_image": segmented_url,
                "num_objects": num_objects
            }
        else:
            buffered = io.BytesIO()
            annotated_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            buffered_seg = io.BytesIO()
            segmented_image.save(buffered_seg, format="JPEG")
            segmented_img_str = base64.b64encode(buffered_seg.getvalue()).decode("utf-8")
            
            print(f"Successfully processed image. Found {num_objects} objects.")
            return {
                "image": f"data:image/jpeg;base64,{img_str}",
                "segmented_image": f"data:image/jpeg;base64,{segmented_img_str}",
                "num_objects": num_objects
            }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    try:
        import torch
        import sys
        print("==========================================================")
        print("Initializing SAM3 RunPod Serverless Worker")
        print("==========================================================")
        print("--- Python/PyTorch Diagnostic ---")
        print("Python version:", sys.version.split()[0])
        print("PyTorch version:", torch.__version__)
        print("CUDA Available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA Device Name:", torch.cuda.get_device_name(0))
            print("CUDA Device Count:", torch.cuda.device_count())
        print("==========================================================")
    except Exception as e:
        print(f"Diagnostic failed: {e}")
        
    runpod.serverless.start({"handler": handler})
