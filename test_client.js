const fs = require('fs');
const path = require('path');

// Helper to manually load .env.local file without external dependencies
function loadEnv() {
  const envPath = path.join(__dirname, '.env.local');
  if (fs.existsSync(envPath)) {
    console.log('Loading environment variables from .env.local...');
    const content = fs.readFileSync(envPath, 'utf8');
    content.split(/\r?\n/).forEach(line => {
      const trimmedLine = line.trim();
      if (trimmedLine && !trimmedLine.startsWith('#')) {
        const parts = trimmedLine.split('=');
        if (parts.length >= 2) {
          const key = parts[0].trim();
          const value = parts.slice(1).join('=').trim();
          process.env[key] = value;
        }
      }
    });
  }
}

// Initialize environment variables
loadEnv();

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;

const inputImagePath = path.join(__dirname, 'input.jpg'); // local test image
const outputImagePath = path.join(__dirname, 'output_client.jpg'); // output path

async function main() {
  if (!RUNPOD_API_KEY || RUNPOD_API_KEY.includes('your_runpod_api_key_here')) {
    throw new Error('Error: RUNPOD_API_KEY is not set or is using the placeholder value. Please set it in .env.local');
  }
  if (!RUNPOD_ENDPOINT_ID || RUNPOD_ENDPOINT_ID.includes('your_runpod_endpoint_id_here')) {
    throw new Error('Error: RUNPOD_ENDPOINT_ID is not set or is using the placeholder value. Please set it in .env.local');
  }

  const url = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`;

  let imageInput;

  // 1. Prepare image input (convert local file to base64, or fallback to test URL if file doesn't exist)
  if (fs.existsSync(inputImagePath)) {
    console.log(`Reading local image from ${inputImagePath} and converting to base64...`);
    const imageBuffer = fs.readFileSync(inputImagePath);
    const base64Image = imageBuffer.toString('base64');
    imageInput = `data:image/jpeg;base64,${base64Image}`;
  } else {
    // Fallback URL so the script runs successfully out-of-the-box
    imageInput = "https://raw.githubusercontent.com/runpod/runpod-python/main/tests/test_images/input.jpg";
    console.log(`Local file 'input.jpg' not found. Using fallback test URL: ${imageInput}`);
  }

  const requestConfig = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${RUNPOD_API_KEY}`
    },
    body: JSON.stringify({
      input: {
        image: imageInput,
        prompt: "eyes"
      }
    })
  };

  console.log(`Sending runsync request to RunPod Endpoint: ${RUNPOD_ENDPOINT_ID}...`);

  try {
    const response = await fetch(url, requestConfig);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // 2. Extract base64 image and convert it back to output image file
    if (data && data.output && data.output.image) {
      let imageBase64 = data.output.image;
      
      // Remove data URL prefix if present
      if (imageBase64.includes(',')) {
        imageBase64 = imageBase64.split(',')[1];
      }

      console.log(`Decoding base64 response and saving output to ${outputImagePath}...`);
      const outputBuffer = Buffer.from(imageBase64, 'base64');
      fs.writeFileSync(outputImagePath, outputBuffer);
      console.log('Success! Result image saved.');
    } else {
      console.warn('Warning: No output image found in response data.', data);
    }

    return data;
  } catch (error) {
    console.error('Error during execution:', error);
    throw error;
  }
}

// Execute the function
main()
  .then(result => {
    console.log('API Response:', JSON.stringify(result, null, 2));
  })
  .catch(error => {
    console.error('Execution Failed:', error.message);
  });
