import os
import pandas as pd
from PIL import Image
import re
from tqdm import tqdm

os.chdir('E:/claude_project/project1_medical_multimodal')
print(os.getcwd())

def load_roco_data(data_path, split='train'):   #['train','val','test']
    '''
    입력 : 데이터 폴더, 추출할 데이터 종류
    출력 : [{이미지 경로,캡션},...]
    '''
    # 데이터 폴더
    csv_names = {
        'train': 'traindata.csv',
        'validation': 'valdata.csv',  # 확인 필요!
        'test': 'testdata.csv'        # 확인 필요!
    }
    base_dir = os.path.join(data_path, split, 'radiology')
    csv_path = os.path.join(base_dir,csv_names.get(split, f'{split}data.csv'))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'{csv_path}를 찾을 수 없습니다.')

    # csv파일 로드
    data_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
    print(f"Loading {split} data...")

    image_dir = os.path.join(base_dir, 'images')
    # csv파일 내 파일명, 캡션 추출
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f'Loading {split} data...'):
        img_file_name = row['name']
        img_file_path = os.path.join(image_dir, img_file_name)

        if not os.path.exists(img_file_path):
            print(f"Warning: Image not found - {img_file_path}")
            continue

        caption = row['caption'].strip()
        if len(caption) < 5:
            continue

        data_list.append({'image_path': img_file_path, 'caption': caption})

    print(f"Loaded {len(data_list)} samples from {split} split")
    return data_list

def preprocess_image(image_path):
    """
        이미지 전처리
        Args:
            image_path: 이미지 파일 경로
        Returns:
            PIL.Image: 이미지
        """
    try:
        image = Image.open(image_path).convert('RGB')
        return image

    except Exception as e:
        raise FileNotFoundError(f'{image_path}가 존재하지 않거나 손상되었습니다: {e}')


def preprocess_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9!?.,'\":;()-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


if __name__ == '__main__':
    data_path = './data/raw/ROCO/all_data'
    data_list = load_roco_data(data_path, split='train')
    text = preprocess_text(data_list[1]['caption'])
    print(text)
