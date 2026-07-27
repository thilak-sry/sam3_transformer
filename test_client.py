import base64
import json
import os
import requests

# Helper to manually load .env.local file without external dependencies
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env.local')
    if os.path.exists(env_path):
        print("Loading environment variables from .env.local...")
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

input_image_path = os.path.join(os.path.dirname(__file__), 'input.jpg')
output_image_path = os.path.join(os.path.dirname(__file__), 'output_client.jpg')

def main():
    if not RUNPOD_API_KEY or "your_runpod_api_key_here" in RUNPOD_API_KEY:
        raise ValueError("Error: RUNPOD_API_KEY is not set or is using placeholder. Set it in .env.local")
    if not RUNPOD_ENDPOINT_ID or "your_runpod_endpoint_id_here" in RUNPOD_ENDPOINT_ID:
        raise ValueError("Error: RUNPOD_ENDPOINT_ID is not set or is using placeholder. Set it in .env.local")

    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"

    # 1. Prepare image input (convert local file to base64, or fallback to test URL if file doesn't exist)
    if os.path.exists(input_image_path):
        print(f"Reading local image from {input_image_path} and converting to base64...")
        with open(input_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        image_input = f"data:image/jpeg;base64,{base64_image}"
    else:
        image_input = "https://raw.githubusercontent.com/runpod/runpod-python/main/tests/test_images/input.jpg"
        print(f"Local file 'input.jpg' not found. Using fallback test URL: {image_input}")

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

    print(f"Sending runsync request to RunPod Endpoint: {RUNPOD_ENDPOINT_ID}...")
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        res_data = response.json()
        
        # 2. Extract base64 image and convert it back to output image file
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
            print("Warning: No output image found in response data.", json.dumps(res_data, indent=2))
            
        return res_data
    except Exception as e:
        print("Error during execution:", e)
        raise e

if __name__ == "__main__":
    main()
