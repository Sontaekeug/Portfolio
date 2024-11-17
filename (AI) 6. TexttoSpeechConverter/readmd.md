# 음성 변환기 (Text-to-Speech Converter)

## 소개
Google의 음성 인식 API(Speech Recognition Engine)와 텍스트 음성 변환 API(gTTS)를 활용하여 음성과 텍스트 간의 양방향 변환을 수행하는 프로그램을 구현하였습니다.

## 기능
- 음성을 텍스트로 변환 (STT)
- 텍스트를 음성으로 변환 (TTS)
- GUI를 통한 텍스트 입력 및 변환된 음성 재생

## 설치 방법
1. 프로젝트를 클론합니다.
   ```bash
   git clone https://github.com/your-repository/voice-converter
   cd voice-converter
2. 라이브러리 설치
- bash : pip install SpeechRecognition gtts pydub pillow
- cmd : python -m pip install SpeechRecognition gtts pydub pillow

## 사용 방법
1. 파일 실행 : python TexttoSpeechConverter.py
2. GUI에서 텍스트 또는 음성을 입력한 후 변환 결과를 확인합니다.

## 주의사항
- Google Speech Recognition API와 Google Text-to-Speech API 사용 시 인터넷 연결이 필요합니다.
- 마이크 및 스피커가 제대로 설정되어 있는지 확인하세요.
- ffmpeg 파일 설치가 필요합니다.
   
