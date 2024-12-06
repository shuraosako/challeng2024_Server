import torch
import os
from flask import Flask, request, send_file, Response
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
from cyclegan.cyclegan import Generator
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = {
   'statue_to_human': 'models/model_generator_AB_246.pth',
   'human_to_statue': 'models/model_generator_BA_246.pth'
}

statue_to_human = Generator().half().to(device)
human_to_statue = Generator().half().to(device)

statue_to_human.load_state_dict(torch.load(MODEL_PATH['statue_to_human'], map_location=device))
human_to_statue.load_state_dict(torch.load(MODEL_PATH['human_to_statue'], map_location=device))

statue_to_human.eval()
human_to_statue.eval()

def preprocess_image(img_array):
    target_size = (512, 512)
    img = Image.fromarray(img_array)
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    print(f"Preprocess input shape: {img_array.shape}")
    print(f"Preprocess input type: {img_array.dtype}")
    print(f"Preprocess input range: {img_array.min()} to {img_array.max()}")
    
    img = torch.from_numpy(img_array.transpose((2, 0, 1))).float()
    print(f"After transpose shape: {img.shape}")
    
    img = img.unsqueeze(0)
    print(f"After unsqueeze shape: {img.shape}")
    
    img = img.half().to(device)
    
    img = img / 255.0
    img = (img - 0.5) / 0.5
    print(f"After normalization range: {img.min().item()} to {img.max().item()}")
    
    return img

def postprocess_image(tensor):
    print(f"Postprocess input shape: {tensor.shape}")
    print(f"Postprocess input range: {tensor.min().item()} to {tensor.max().item()}")

    img = (tensor + 1) / 2
    img = img.squeeze().detach().cpu().numpy()
    print(f"After denormalization shape: {img.shape}")

    # ここで画像を左に90度回転
    img = np.rot90(img, 1, (0, 1))  # 1は回転回数、(0, 1)は回転させる軸

    img = img.transpose(1, 2, 0) * 255.0
    img = np.clip(img, 0, 255)
    print(f"Final shape: {img.shape}")
    print(f"Final range: {img.min()} to {img.max()}")

    return img

@app.route('/transform', methods=['POST'])
def transform_image():
    try:
        print("1. リクエスト受信")
        raw_data = request.get_data()
        content_type = request.headers.get('Content-Type', '')
        print(f"2. Content-Type: {content_type}")
        
        if not content_type.startswith('multipart/form-data'):
            print("3. Invalid Content-Type")
            return 'Invalid Content-Type', 400
            
        boundary = content_type.split('boundary=')[-1].encode()
        print(f"4. Boundary: {boundary}")
        parts = raw_data.split(boundary)
        print(f"5. Parts count: {len(parts)}")
        
        for i, part in enumerate(parts):
            print(f"6. Processing part {i}")
            if b'filename' in part and b'image.jpg' in part:
                print("7. Found image part")
                idx = part.find(b'\r\n\r\n')
                if idx == -1:
                    print("8. No data boundary found")
                    continue
                    
                image_data = part[idx+4:]
                if image_data.endswith(b'\r\n'):
                    image_data = image_data[:-2]
                
                print("9. Opening image data")
                img = Image.open(io.BytesIO(image_data))
                print(f"10. Image mode: {img.mode}")
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                print("11. Converting to array")
                img_array = np.array(img)
                print(f"12. Array shape: {img_array.shape}")
                
                print("13. Preprocessing image")
                img_tensor = preprocess_image(img_array)
                print(f"14. Tensor shape: {img_tensor.shape}")
                
                print("15. Running model inference")
                with torch.no_grad():
                    transformed_tensor = statue_to_human(img_tensor)
                    transformed_tensor = transformed_tensor.float().cpu()
                print(f"16. Transformed tensor shape: {transformed_tensor.shape}")
                
                print("17. Postprocessing image")
                transformed_array = postprocess_image(transformed_tensor)
                transformed_img = Image.fromarray(transformed_array.astype('uint8'))
                
                print("18. Preparing response")
                img_byte_arr = io.BytesIO()
                transformed_img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                print("19. Sending response")
                response = send_file(
                    img_byte_arr,
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='transformed.png'
                )
                response.headers['Access-Control-Allow-Origin'] = '*'
                print("20. Response sent successfully")
                return response
                
        print("21. No image found in request")
        return 'No image found in request', 400
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return str(e), 500

@app.route('/health', methods=['GET'])
def health_check():
   return 'OK', 200

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=50000, debug=True)