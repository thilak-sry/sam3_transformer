import base64
import io
import requests
from PIL import Image
import runpod
from inference import run_inference

def handler(event):
    print("Worker Start")
    input_data = event['input']
    
    image_input = input_data.get('image', './0.jpg')
    prompt = input_data.get('prompt', 'eyes')  

    print(f"Received prompt: {prompt}")
    
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
    annotated_image, num_objects = run_inference(image, prompt)
    
    # Save the output image locally
    output_path = "./output.jpg"
    annotated_image.save(output_path)
    print(f"Saved output image locally to {output_path}")
    
    # Convert output image to base64
    buffered = io.BytesIO()
    annotated_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    print(f"Successfully processed image. Found {num_objects} objects.")
    
    return {
        "image": f"data:image/jpeg;base64,{img_str}",
        "num_objects": num_objects
    }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})