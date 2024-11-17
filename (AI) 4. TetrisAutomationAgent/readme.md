# 테트리스 자동화 에이전트 (Tetris Automation Agent)

## 소개
컨볼루션 신경망(CNN)과 딥 Q-러닝을 결합한 강화학습 알고리즘을 기반으로, 테트리스 보드 패턴 인식, 상태 평가를 통해 의사결정을 수행, 경험 재생(라운드 별 게임 결과 지속 학습)을 통한 테트리스 게임을 자동으로 플레이하는 AI 시스템입니다.

## 기능
- 테트리스 게임 자동 플레이
- 보상 시스템을 통해 최적의 움직임 학습
- 다양한 테트리스 환경에 대응 가능

## 설치 방법
1. 프로젝트를 클론합니다.
   ```bash
   git clone https://github.com/your-repository/tetris-agent
   cd tetris-agent
2. 라이브러리 설치
- bash : pip install gym tensorflow pygame numpy
- cmd : python -m pip install gym tensorflow pygame numpy

## 사용 방법
1. 파일 실행 : python TetrisAutomationAgent.py
2. 실행 후 AI 에이전트가 자동으로 테트리스 게임을 플레이하는 모습을 확인할 수 있습니다.

## 주의사항
- 강화 학습 모델 훈련에는 시간이 소요되며, GPU 환경에서 실행하는 것을 추천합니다.
- 여러 라운드 동안 학습 후 성능이 향상됩니다.
  * 상당히 많은 라운드(10,000) 이상부터 효과가 있으므로 참고바랍니다.
