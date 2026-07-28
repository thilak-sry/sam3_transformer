import base64
import json
import os
import requests

# Helper to manually load .env.local file without external dependencies
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env.local')
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(__file__), 'env.local')
    if os.path.exists(env_path):
        print(f"Loading environment variables from {os.path.basename(env_path)}...")
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                trimmed = line.strip()
                if trimmed and not trimmed.startswith('#'):
                    if '=' in trimmed:
                        key, value = trimmed.split('=', 1)
                        os.environ[key.strip()] = value.strip()

# Initialize environment variables
load_env()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

input_image_path = os.path.join(os.path.dirname(__file__), 'face.jpg')
output_image_path = os.path.join(os.path.dirname(__file__), 'output_client.jpg')
output_segmented_image_path = os.path.join(os.path.dirname(__file__), 'output_segmented_client.jpg')

def main():
    if not RUNPOD_API_KEY or "your_runpod_api_key_here" in RUNPOD_API_KEY:
        raise ValueError("Error: RUNPOD_API_KEY is not set or is using placeholder. Set it in .env.local")
    if not RUNPOD_ENDPOINT_ID or "your_runpod_endpoint_id_here" in RUNPOD_ENDPOINT_ID:
        raise ValueError("Error: RUNPOD_ENDPOINT_ID is not set or is using placeholder. Set it in .env.local")

    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"

    # 1. Prepare image input (convert local file to base64)
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Error: Local test image '{input_image_path}' not found. Please place an image at that path to run the test.")
        
    print(f"Reading local image from {input_image_path} and converting to base64...")
    with open(input_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    image_input = f"data:image/jpeg;base64,{base64_image}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }

    data = {
        "input": {
            "image": image_input,
            "prompt": "eyes"
        }
    }

    print(f"Sending request to RunPod Endpoint: {RUNPOD_ENDPOINT_ID}...")
    try:
        response = requests.post(url, headers=headers, json=data, timeout=600)
        response.raise_for_status()
        res_data = response.json()
        
        job_id = res_data.get("id")
        status = res_data.get("status")
        
        # If the job is still in queue or progress, poll for completion
        if status in ["IN_QUEUE", "IN_PROGRESS"]:
            import time
            print(f"Job is {status} (ID: {job_id}). Polling for results...")
            status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
            
            while status in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(3)
                try:
                    status_response = requests.get(status_url, headers=headers, timeout=10)
                    status_response.raise_for_status()
                    res_data = status_response.json()
                    status = res_data.get("status")
                    print(f"Current Job Status: {status}")
                except Exception as poll_error:
                    print(f"Polling warning: {poll_error}. Retrying...")
                    
        # 2. Extract base64 image and convert it back to output image file
        if status == "COMPLETED":
            output_data = res_data.get("output", {})
            if output_data and "image" in output_data:
                image_str = output_data["image"]
                if "," in image_str:
                    image_str = image_str.split(",", 1)[1]
                
                print(f"Decoding base64 response and saving output to {output_image_path}...")
                image_bytes = base64.b64decode(image_str)
                with open(output_image_path, "wb") as f:
                    f.write(image_bytes)
                print("Success! Result image saved.")
            else:
                print("Warning: No output image found in completed response data.", json.dumps(res_data, indent=2))
                
            if output_data and "segmented_image" in output_data:
                seg_image_str = output_data["segmented_image"]
                if "," in seg_image_str:
                    seg_image_str = seg_image_str.split(",", 1)[1]
                
                print(f"Decoding base64 response and saving segmented output to {output_segmented_image_path}...")
                seg_image_bytes = base64.b64decode(seg_image_str)
                with open(output_segmented_image_path, "wb") as f:
                    f.write(seg_image_bytes)
                print("Success! Segmented result image saved.")
        else:
            print(f"Job finished with status: {status}")
            print(json.dumps(res_data, indent=2))
            
        return res_data
    except Exception as e:
        print("Error during execution:", e)
        raise e

if __name__ == "__main__":
    main()
