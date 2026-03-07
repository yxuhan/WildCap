import os
import argparse
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_root', type=str, default="xxx")
parser.add_argument('--save_root', type=str, default="xxx")
parser.add_argument('--threads', type=int, default=3)
parser.add_argument('--api', type=str)
opt, _ = parser.parse_known_args()

os.makedirs(opt.save_root, exist_ok=True)

API_URL = "https://sdk.beeble.ai/v1/acquire/albedo"
API_KEY = opt.api

session = requests.Session()
session.headers.update({
    'x-api-key': API_KEY,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Origin': 'https://beeble.ai',
    'Referer': 'https://beeble.ai/'
})

def process_image(file_name):
    img_pth = os.path.join(opt.img_root, file_name)
    save_pth = os.path.join(opt.save_root, file_name)
    
    if os.path.exists(save_pth):
        return f"Skipped: {file_name}"

    try:
        time.sleep(random.uniform(0.1, 0.5)) 
        
        with open(img_pth, 'rb') as f:
            files = {'source_image': (file_name, f, 'image/jpeg')}
            data = {'preview': 'false'}
            
            response = session.post(API_URL, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                with open(save_pth, 'wb') as out_f:
                    out_f.write(response.content)
                return f"Success: {file_name}"
            elif response.status_code == 429:
                return f"Error: Rate limited (429) on {file_name}. Try reducing threads."
            else:
                return f"Error {response.status_code}: {file_name} - {response.text[:100]}"
                
    except Exception as e:
        return f"Failed: {file_name} due to {str(e)}"

def main():
    img_list = sorted([f for f in os.listdir(opt.img_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Total images to process: {len(img_list)}")

    with ThreadPoolExecutor(max_workers=opt.threads) as executor:
        results = list(executor.map(process_image, img_list))

    for res in results:
        print(res)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\n--- Total time spent: {time.time() - start_time:.2f} seconds ---")
