## 사용 알고리즘
Word2Vec과 FastText를 이용한 플레이리스트 예측 모델입니다.   

***

### 필요한 데이터
* 해당 모델 재현 시 필요한 데이터는 아래와 같습니다. 데이터를 ./dataset 폴더에 위치시켜주세요.   
  4번의 경우 ./dataset에 위치한 most_popular_res.json을 사용해주시면 됩니다.   
1. train.json
2. test.json
3. song_meta.json
4. most_popular_res.json 

***

### 모델 다운로드
* 학습된 Word2vec 모델은 아래 url에서 다운받으실 수 있습니다.   
  다운받은 모델은 압축해제하여 inference.py가 있는 곳에 함께 위치시켜주세요.   
[Word2vec 모델 다운로드](https://arenamelon2.blogspot.com/2020/07/kakao-arena-melon-playlist-continuation.html)   

* 학습된 FastText 모델의 경우 별도 첨부하진 않았습니다.   

***

### 전체 폴더구조
모델 다운로드까지 완료되고 나면 전체 폴더구조는 아래와 같아야 합니다.   
```
.
├── arena_melon
│   ├── dataset
│   │    ├── train.json
│   │    ├── test.json
│   │    ├── song_meta.json
│   │    └── most_popular_res.json
│   ├── train
│   │    └── train.py
│   ├── PlaylistEmbedding.py
│   ├── inference.py
│   ├── w2v_model.model.trainables.syn1neg.npy
│   ├── w2v_model.model.wv.vectors.npy
│   └── w2v_model.model
```


***


### 예측결과 재현
arena_melon 폴더로 위치를 이동한 후 아래 코드를 실행시켜주세요.   
예측 결과 파일은 ~~ 위치에 만들어집니다.    
```
python inference.py 
```

***


### 모델학습 재현
arena_melon/train 위치로 이동 후 아래 코드를 실행시켜주세요.   
이때 arena_melon/train 폴더에 "w2v_model.model", "FT_title_model.model" 이름의 학습된 모델 파일이 있을 경우 모델 학습 재현이 되지 않는 점 참고 부탁드립니다. 
```
python train.py
```
위 코드를 이용해 새롭게 생성된 모델로 예측 결과 재현이 필요하신 경우 train.json을 제외한
모든 파일을 arena_melon 폴더로 옮긴 후 python inference.py를 실행시켜주세요.

