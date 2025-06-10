
# IMU based activity classifier

## 개요
이 프로젝트는 첨단분야 혁신융합대학 차세대통신 사업단, 국민대학교와 울산과학대학교가 참여한 반려견을 위한 헬스케어 시스템 시제품 제작이 목적입니다.

## 프로젝트 구조
```
.
├── dataset/
│   ├── DogInfo.csv         # 반려견 메타 정보 (ID, 품종 등)
│   └── DogMoveData.csv     # 센서 데이터와 행동 라벨
├── Pet\_i\_Behavior\_Classification.ipynb  # 모듈화된 Jupyter Notebook
├── main.py                 # 모듈화 이전의 스크립트 버전 (참고용)
├── README.md               # 프로젝트 설명서
└── requirements.txt        # 프로젝트 의존성 목록

````

## 요구 사항 (Requirements)
- Python 3.8+
- MacOS: TensorFlow 2.6+
- scikit-learn 1.x  
- pandas 1.x  
- numpy 1.x  
- matplotlib 3.x  

```bash
# 의존성 설치 (requirements.txt 파일 활용)
pip install -r requirements.txt
````
