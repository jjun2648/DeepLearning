# 추천 모델 구현
- 주제 : 주어진 DeepFM 라이브러리를 이용하여 영화 추천 모델 만들기
    - userId가 title인 영화를 볼 확률을 구하는 모델을 만드세요.
    - 해당 과제의 경우 모델 훈련 뿐만 아닌, train_test_split으로 분기한 test 데이터의 pred_y 값을 [userId, title, pred] 세개의 열을 보유한 csv 파일로 저장하는 것까지 해주세요.
- 데이터
    
    https://drive.google.com/file/d/1KcR9KFGRdsIv2PrfxhiS0rS7nyTJy9-3/view?usp=drive_link\

- 실행 방법
    - 파일 경로에서 python main.py 입력
      ```PowerShell
      python main.py
      ```

    - 결과는 data 폴더에 csv파일로 저장
