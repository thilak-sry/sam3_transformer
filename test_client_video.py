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
                        val = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = val

# Initialize environment variables
load_env()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

def main():
    if not RUNPOD_API_KEY or "your_runpod_api_key_here" in RUNPOD_API_KEY:
        raise ValueError("Error: RUNPOD_API_KEY is not set or is using placeholder. Set it in .env.local")
    if not RUNPOD_ENDPOINT_ID or "your_runpod_endpoint_id_here" in RUNPOD_ENDPOINT_ID:
        raise ValueError("Error: RUNPOD_ENDPOINT_ID is not set or is using placeholder. Set it in .env.local")

    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"

    # Prepare input payload using the Azure SAS video URL
    video_url = "https://srystudiostorage.blob.core.windows.net/segment-anything/266435_medium.mp4?sp=r&st=2026-07-28T09:59:07Z&se=2027-12-01T18:14:07Z&spr=https&sv=2026-02-06&sr=b&sig=0bABg9SakkokUvvOx00tddH0AB4G9g0WYlKDSBU1sig%3D"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }

    data = {
        "input": {
            "image": video_url,
            "prompt": "upper lips"
        }
    }

    print(f"Sending video request to RunPod Endpoint: {RUNPOD_ENDPOINT_ID}...")
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
                    
        # Extract output files (images or videos)
        if status == "COMPLETED":
            output_data = res_data.get("output", {})
            print("\n================ Job Output ================")
            
            for key, default_filename in [
                ("image", "output_client.jpg"), 
                ("segmented_image", "output_segmented_client.jpg"),
                ("video", "output_client.mp4"),
                ("segmented_video", "output_segmented_client.mp4")
            ]:
                if output_data and key in output_data:
                    val = output_data[key]
                    if val.startswith("http"):
                        print(f"Result {key} URL: {val}")
                    else:
                        # Decode base64
                        if "," in val:
                            val = val.split(",", 1)[1]
                        filepath = os.path.join(os.path.dirname(__file__), default_filename)
                        print(f"Decoding base64 response and saving {key} to {filepath}...")
                        file_bytes = base64.b64decode(val)
                        with open(filepath, "wb") as f:
                            f.write(file_bytes)
                        print(f"Success! {key} saved.")
            print("============================================")
        else:
            print(f"Job finished with status: {status}")
            print(json.dumps(res_data, indent=2))
            
        return res_data
    except Exception as e:
        print("Error during execution:", e)
        raise e

if __name__ == "__main__":
    main()
