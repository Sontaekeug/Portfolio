# 텍스트 요약기 (Text Summarizer)

## 소개
KoBART 모델을 기반으로 한 텍스트 요약 프로그램으로, 트랜스포머 기반의 인코더-디코더 구조를 활용해 한국어 텍스트의 핵심 내용을 추출하는 프로그램입니다.('gogamza/kobart-summarization' 사전학습 모델을 활용하여 문맥을 파악)

## 기능
- 한국어 텍스트 요약
- 요약 결과를 GUI에서 바로 확인 가능

## 설치 방법
1. 프로젝트를 클론합니다.
   ```bash
   git clone https://github.com/your-repository/text-summarizer
   cd text-summarizer
2. 라이브러리 설치
- bash : pip install torch transformers pyqt5
- cmd : python -m pip install torch transformers pyqt5

## 사용 방법
1. 파일 실행 : python TextSummarizer.py
2. GUI에서 텍스트를 입력하고 요약 버튼을 눌러 요약 결과를 확인합니다.

## 주의사항
- gogamza/kobart-summarization 모델이 필요하며, 처음 실행 시 인터넷을 통해 모델 파일을 다운로드해야 합니다.
- 긴 텍스트를 입력할 때 처리 시간이 길어질 수 있습니다.
- KoBART 모델의 경우 요약보다 핵심 내용 추출에 초점이 맞춰진 라이브러리로, GPT / 클로드와 같은 LLM AI들의 자연어 처리 수준과 다름에 대한 인지가 필요합니다.
