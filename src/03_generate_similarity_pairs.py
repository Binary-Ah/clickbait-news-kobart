import os
import glob
import json
import random
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from tqdm.auto import tqdm

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    "model_path": "./results/final_model",
    
    # 데이터 경로 (라벨링데이터 폴더)
    "data_dir": "data/146.낚시성_기사_탐지_데이터/01-1.정식개방데이터/Training/02.라벨링데이터",
    
    # 타겟 패턴 (문자열 확인 필수)
    "target_patterns": ["13", "14", "15", "16"], 
    "target_count": 17375,
    
    "batch_size": 32,      
    "max_input_len": 512,  
    "max_summary_len": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 2. 데이터 수집 함수 (제목 추출 추가)
# ==========================================
def collect_data():
    print(f"📂 데이터 수집 시작: {CONFIG['data_dir']}")
    
    search_pattern = os.path.join(CONFIG['data_dir'], "**", "*.json")
    all_json_files = glob.glob(search_pattern, recursive=True)
    
    if not all_json_files:
        print("❌ JSON 파일을 찾지 못했습니다.")
        return []

    print(f"   -> 총 발견된 파일: {len(all_json_files)}개")
    random.shuffle(all_json_files) 
    
    clickbait_data = []
    normal_data = []
    
    print("   -> 파일 분석 및 로드 중...")
    
    for f in tqdm(all_json_files, desc="Scanning"):
        if len(clickbait_data) >= CONFIG['target_count'] and len(normal_data) >= CONFIG['target_count']:
            break
            
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                source_info = data.get('sourceDataInfo', {})
                
                # [수정 포인트] 패턴 확인 및 제목 추출
                pattern = str(source_info.get('processPattern', "00"))
                title = source_info.get('newsTitle', "") # 제목 추출
                content = source_info.get('newsContent', "") # 본문 추출
                
                if not content: continue

                # 공통 데이터 딕셔너리 생성
                data_item = {
                    "title": title,           # 제목 추가
                    "original_text": content, # 본문
                    "pattern": pattern
                }

                # A. 낚시성 기사 (13~16)
                if pattern in CONFIG['target_patterns']:
                    if len(clickbait_data) < CONFIG['target_count']:
                        data_item["label"] = 1
                        clickbait_data.append(data_item)
                
                # B. 일반 기사 (00)
                elif pattern == "00" or pattern == "0":
                    if len(normal_data) < CONFIG['target_count']:
                        data_item["label"] = 0
                        normal_data.append(data_item)
                    
        except Exception:
            continue
            
    print(f"   => 낚시성(패턴 13~16) 확보: {len(clickbait_data)}개")
    print(f"   => 일반기사(패턴 00) 확보: {len(normal_data)}개")
    
    return clickbait_data + normal_data

# ==========================================
# 3. 전처리 및 데이터셋 클래스
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class InferenceDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ==========================================
# 4. 메인 실행
# ==========================================
def main():
    # 1) 데이터 수집
    raw_data = collect_data()
    
    if not raw_data:
        print("❌ 데이터를 수집하지 못했습니다.")
        return

    print(f"\n🚀 총 {len(raw_data)}개 데이터에 대해 요약 생성 시작...")
    
    # 2) 모델 로드
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(CONFIG['model_path'])
        model = BartForConditionalGeneration.from_pretrained(CONFIG['model_path'])
        model.to(CONFIG['device'])
        model.eval()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 3) DataLoader
    dataset = InferenceDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    results = []
    
    # 4) 요약 생성
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            # 배치에서 데이터 추출
            titles = batch['title']
            texts = batch['original_text']
            labels = batch['label']
            patterns = batch['pattern']
            
            # 본문 전처리 & 토큰화
            cleaned_texts = [clean_text(t) for t in texts]
            inputs = tokenizer(
                cleaned_texts,
                max_length=CONFIG['max_input_len'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(CONFIG['device'])
            
            # 요약 생성
            summary_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=CONFIG['max_summary_len'],
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            # 디코딩
            summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            
            # 결과 저장 (제목 포함)
            for i in range(len(texts)):
                results.append({
                    "title": titles[i],               # 제목
                    "generated_summary": summaries[i], # 요약문
                    "original_content": texts[i],      # 원문
                    "label": labels[i].item(),         # 라벨
                    "process_pattern": patterns[i]     # 패턴
                })

    # 5) 저장
    df = pd.DataFrame(results)
    
    # 컬럼 순서 보기 좋게 정렬
    df = df[['title', 'generated_summary', 'label', 'process_pattern', 'original_content']]
    
    # 셔플 및 저장
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    save_filename = "clickbait_dataset_with_title.csv"
    df.to_csv(save_filename, index=False)
    
    print(f"\n✨ 완료! 저장 파일: {save_filename}")
    print(f"   - 총 개수: {len(df)}")
    print(f"   - 낚시성(1): {len(df[df['label']==1])}")
    print(f"   - 일반(0): {len(df[df['label']==0])}")

if __name__ == "__main__":
    main()