import os
import glob
import json
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerFast, 
    BartForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from tqdm.auto import tqdm

# ==========================================
# 1. RTX 4090 최적화 설정 (Configuration)
# ==========================================
CONFIG = {
    "model_name": "gogamza/kobart-summarization",
    "data_dir": "data/022.요약문_및_레포트_생성_데이터", 
    "cache_dir": "./cache_data",    
    "output_dir": "./results",      
    "max_input_len": 512,           
    "max_target_len": 128,          
    
    # [4090 최적화] 배치 사이즈 UP
    "batch_size": 32,                
    "num_epochs": 5,                
    "learning_rate": 2e-5,          
    "num_workers": 0,               # Windows 에러 방지
    "seed": 42,
    
    "warmup_steps": 1000,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3 
}

torch.manual_seed(CONFIG['seed'])
if not os.path.exists(CONFIG['cache_dir']):
    os.makedirs(CONFIG['cache_dir'])

# ==========================================
# 2. 텍스트 정제 함수
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\([^)]+[=].+?[=]', '', text)
    text = re.sub(r'\([^)]+\) *[^)]+ 기자 =', '', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    text = re.sub(r'\[(사진|자료)(제공)?=.*?\]', '', text)
    text = re.sub(r'무단 ?전재.*?금지', '', text)
    text = re.sub(r'ⓒ.*?reserved\.', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 3. 데이터 로드 및 전처리
# ==========================================
def load_and_preprocess(split_keyword):
    cache_file = os.path.join(CONFIG['cache_dir'], f"{split_keyword}.csv")
    
    if os.path.exists(cache_file):
        print(f"🚀 캐시된 데이터 로드 중 ({split_keyword}): {cache_file}")
        return pd.read_csv(cache_file).dropna()

    print(f"📂 원본 데이터 탐색 중... 키워드: '{split_keyword}'")
    search_pattern = os.path.join(CONFIG['data_dir'], "**", "*.json")
    all_json_files = glob.glob(search_pattern, recursive=True)
    target_files = [f for f in all_json_files if split_keyword in f]
    
    if not target_files:
        print(f"⚠️ 파일 없음: {split_keyword}")
        return pd.DataFrame()
    
    print(f"   -> {len(target_files)}개 파일 처리 중...")
    data_list = []
    for f in tqdm(target_files):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list): data_list.extend(data)
                else: data_list.append(data)
        except: continue
            
    if not data_list: return pd.DataFrame()

    df = pd.DataFrame(data_list)
    try:
        if 'Meta(Refine)' in df.columns:
            meta = pd.json_normalize(df['Meta(Refine)'])
            if 'passage' in meta.columns: df['passage'] = meta['passage']
        
        if 'Annotation' in df.columns:
            df['summary1'] = df['Annotation'].apply(
                lambda x: x['summary1'] if isinstance(x, dict) else (x[0]['summary1'] if isinstance(x, list) and len(x)>0 else "")
            )
    except: pass

    if 'passage' in df.columns and 'summary1' in df.columns:
        final_df = df[['passage', 'summary1']].dropna()
        final_df = final_df[final_df['passage'].str.len() > 10]
        final_df.to_csv(cache_file, index=False)
        return final_df
    else:
        return pd.DataFrame()

# ==========================================
# 4. Dataset 정의
# ==========================================
class SummaryDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            clean_text(str(row['passage'])),
            max_length=CONFIG['max_input_len'],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        targets = self.tokenizer(
            text_target=clean_text(str(row['summary1'])),
            max_length=CONFIG['max_target_len'],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs['input_ids'].squeeze(),
            "attention_mask": inputs['attention_mask'].squeeze(),
            "labels": labels
        }

# ==========================================
# 5. 메인 실행
# ==========================================
if __name__ == "__main__":
    print(f"🔥 RTX 4090 최적화 학습 시작 (Batch: {CONFIG['batch_size']}, bf16: True)")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(CONFIG['model_name'])
    model = BartForConditionalGeneration.from_pretrained(CONFIG['model_name'])
    
    train_df = load_and_preprocess("Training")
    val_df = load_and_preprocess("Validation")
    
    if train_df.empty or val_df.empty:
        print("❌ 데이터 로드 실패"); exit()

    train_dataset = SummaryDataset(train_df, tokenizer)
    eval_dataset = SummaryDataset(val_df, tokenizer)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG['output_dir'],
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        num_train_epochs=CONFIG['num_epochs'],
        
        # [4090 핵심 설정]
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=1,
        bf16=True, # 4090 가속 핵심
        fp16=False,
        
        # [속도 핵심] 평가는 에폭 단위로
        eval_strategy="epoch", 
        save_strategy="epoch",
        
        learning_rate=CONFIG['learning_rate'],
        weight_decay=0.01,
        warmup_steps=CONFIG['warmup_steps'],
        max_grad_norm=CONFIG['max_grad_norm'],
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=CONFIG['max_target_len'],
        dataloader_num_workers=CONFIG['num_workers'],
        logging_dir='./logs',
        logging_steps=50,
        report_to="none"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['early_stopping_patience'])]
    )
    
    trainer.train(resume_from_checkpoint=True)
    
    model.save_pretrained(os.path.join(CONFIG['output_dir'], "final_model"))
    tokenizer.save_pretrained(os.path.join(CONFIG['output_dir'], "final_model"))
    print("✨ 학습 완료!")