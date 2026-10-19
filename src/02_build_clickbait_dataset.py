import os
import glob
import json
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from tqdm.auto import tqdm
import re

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    # 1) 학습시킨 요약 모델 경로 (체크포인트)
    "model_path": "./results/checkpoint-42500", 
    
    # 2) 낚시성 기사 데이터 최상위 경로
    "base_data_dir": "data/146.낚시성_기사_탐지_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/",
    
    # 3) 데이터 추출 설정
    "target_patterns": [13, 14, 15, 16], # 낚시성 기사 패턴
    "target_count": 17375,               # 각 클래스별 목표 개수
    
    # 4) 요약 생성 설정 (4090 기준)
    "batch_size": 32,      # 요약 생성 시 배치 크기
    "max_input_len": 512,  # 모델 입력 최대 길이
    "max_summary_len": 128 # 생성될 요약문 최대 길이
}

# ==========================================
# 2. 데이터 수집 함수 (JSON 파싱)
# ==========================================
def collect_raw_data():
    print("🚀 데이터 수집 시작...")
    
    # -------------------------------------------------
    # A. 낚시성 기사 (Clickbait_Direct) 추출
    # -------------------------------------------------
    print("   [1/2] 낚시성 기사(Direct) 탐색 중...")
    
    # '라벨링데이터' 폴더 아래 'Direct'가 포함된 경로 찾기
    clickbait_files = glob.glob(os.path.join(CONFIG['base_data_dir'], "**", "*Direct*.json"), recursive=True)
    
    clickbait_raw = []
    
    for f in tqdm(clickbait_files, desc="Clickbait Filtering"):
        if len(clickbait_raw) >= CONFIG['target_count']:
            break
            
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # 메타데이터 확인 (labeledDataInfo -> processPattern)
                info = data.get('labeledDataInfo', {})
                pattern = info.get('processPattern', 0)
                
                if pattern in CONFIG['target_patterns']:
                    # 본문 내용 추출 (newsContent)
                    source_info = data.get('sourceDataInfo', {})
                    content = source_info.get('newsContent', "")
                    
                    if content:
                        clickbait_raw.append({
                            "original_text": content,
                            "label": 1  # 낚시성
                        })
        except:
            continue
            
    print(f"   => 낚시성 기사 {len(clickbait_raw)}개 확보 완료.")

    # -------------------------------------------------
    # B. 일반 기사 (Non_Clickbait) 추출
    # -------------------------------------------------
    print("   [2/2] 일반 기사(Non_Clickbait) 탐색 중...")
    
    # '일반기사' 또는 'Non_Clickbait'가 포함된 파일 찾기
    # 보통 파일명이나 폴더명에 'Non_Clickbait' 혹은 '일반기사'가 들어감
    # 정확도를 위해 'Clickbait' 글자가 없는 파일들을 일반 기사로 간주하거나 특정 폴더 지정 필요
    # 여기서는 '일반기사' 폴더를 찾는 패턴 사용
    normal_files = glob.glob(os.path.join(CONFIG['base_data_dir'], "**", "*일반기사*.json"), recursive=True)
    
    # 만약 위 패턴으로 안 잡히면 'Non_Clickbait'으로 시도
    if not normal_files:
        normal_files = glob.glob(os.path.join(CONFIG['base_data_dir'], "**", "*Non_Clickbait*.json"), recursive=True)

    # 랜덤 샘플링
    if len(normal_files) > CONFIG['target_count']:
        selected_files = random.sample(normal_files, CONFIG['target_count'])
    else:
        selected_files = normal_files
        
    normal_raw = []
    for f in tqdm(selected_files, desc="Normal News Loading"):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                source_info = data.get('sourceDataInfo', {})
                content = source_info.get('newsContent', "")
                
                if content:
                    normal_raw.append({
                        "original_text": content,
                        "label": 0  # 일반 기사
                    })
        except:
            continue
            
    print(f"   => 일반 기사 {len(normal_raw)}개 확보 완료.")
    
    return clickbait_raw + normal_raw

# ==========================================
# 3. 요약 생성용 Dataset & DataLoader
# ==========================================
class SummarizationDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 4. 메인 프로세스
# ==========================================
def main():
    # 1) 원본 데이터(본문) 수집
    raw_data = collect_raw_data()
    
    if not raw_data:
        print("❌ 데이터를 찾지 못했습니다. 경로를 확인해주세요.")
        return

    print(f"\n🚀 총 {len(raw_data)}개 기사에 대해 요약 생성 시작 (Model: {CONFIG['model_path']})")
    
    # 2) 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(CONFIG['model_path'])
        model = BartForConditionalGeneration.from_pretrained(CONFIG['model_path']).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 3) 배치 단위 요약 생성
    dataset = SummarizationDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    final_dataset = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Summaries"):
            texts = batch['original_text']
            labels = batch['label']
            
            # 전처리
            cleaned_texts = [clean_text(t) for t in texts]
            
            # 토큰화
            inputs = tokenizer(
                cleaned_texts,
                max_length=CONFIG['max_input_len'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # 생성
            summary_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=CONFIG['max_summary_len'],
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # 디코딩
            summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            
            # 결과 저장
            for summary, label in zip(summaries, labels):
                final_dataset.append({
                    "summary": summary,
                    "label": label.item()
                })
                
    # 4) 최종 저장
    df = pd.DataFrame(final_dataset)
    save_path = "train_clickbait_summary.csv"
    df.to_csv(save_path, index=False)
    
    print("\n✅ 작업 완료!")
    print(f"   - 저장 파일: {save_path}")
    print(f"   - 총 데이터 수: {len(df)}")
    print(f"   - 낚시성(1) 개수: {len(df[df['label']==1])}")
    print(f"   - 일반(0) 개수: {len(df[df['label']==0])}")

if __name__ == "__main__":
    main()