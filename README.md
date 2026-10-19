# 낚시성(Clickbait) 기사 판별을 위한 KoBART 요약 모듈 & 요약 모델 선정 실험

2025-2 자연어처리 수업 팀 프로젝트 (팀 레포 [jinyoung924/ssu_25-2_NLP](https://github.com/jinyoung924/ssu_25-2_NLP)) · 이 레포는 **내가 맡은 요약 모듈 파트**만 분리한 것

## Problem
헤드라인과 본문 내용이 다른 낚시성 기사를 잡으려면, 본문을 짧게 요약한 뒤 헤드라인과의 의미 유사도를 재야 한다. 그 **요약기**를 한국어 뉴스에 맞게 파인튜닝하고, 어떤 요약 모델을 쓸지 비교 실험했다.

## Pipeline
```
AI Hub 022.요약문·레포트 생성 데이터 ─► 01 KoBART 파인튜닝 (gogamza/kobart-summarization)
AI Hub 146.낚시성 기사 탐지 데이터 ────► 02 패턴 13~16 낚시성 기사 + 정상 기사 균형 추출 (각 17,375건) → 요약 생성
                                       ─► 03 (헤드라인, 요약) 쌍 생성 → 유사도 모델(KoSimCSE 등, 팀원 파트) 학습 데이터
```
- 01: 기자명·이메일·저작권 문구 정규식 제거, 입력 512 / 요약 128 토큰, batch 32, 5 epoch, lr 2e-5, warmup 1000, EarlyStopping(patience 3), seed 42 — RTX 4090 기준 설정
- 모델 선정 실험(`notebooks/`): KoBART vs **Llama-3 (LoRA, 4-bit)** vs **Polyglot-5.8B (LoRA)** — 1,000건 샘플로 생성 성공률·ROUGE 비교, Llama용 프롬프트 수정 전후 비교

## Results (`results/`)
| 모델 | 생성 성공률 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| Llama-3 (LoRA/Chat) | 100% | 25.28 | 4.07 | 24.78 |
| Polyglot-5.8B (LoRA) | 100% | 19.26 | 3.89 | 18.83 |
| Baseline BART (무조정) | 0% | – | – | – |

무조정 BART는 한국어 요약을 전혀 만들지 못했고, LoRA 적용 LLM 중 Llama-3가 ROUGE가 높았다. 최종 서비스에는 속도·메모리를 고려해 파인튜닝한 KoBART를 요약 모듈로 사용. 팀 결과(유사도 임계값 0.940, F1 ≈ 0.83)는 팀 레포 참고.

## Layout
```
src/01_train_kobart_summarizer.py
src/02_build_clickbait_dataset.py
src/03_generate_similarity_pairs.py
notebooks/model_selection_experiments.ipynb   # LoRA 학습 + 평가
notebooks/model_selection_eval_only.ipynb     # 학습 없이 평가·시각화
results/final_evaluation_result.csv, final_lora_evaluation_result.csv
```
AI Hub 데이터·체크포인트(`results/checkpoint-*`, LoRA 가중치)는 포함하지 않았다. 데이터 경로는 `data/…` 로 바꿔 두었으니 본인 환경에 맞게 수정.

## My role
요약 모듈 담당 — KoBART 파인튜닝, 낚시성 데이터 추출·요약 생성 파이프라인, 요약 모델 선정 실험(LoRA 비교). 유사도 모델·서비스 UI는 팀원.
