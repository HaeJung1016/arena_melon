# -*- coding: utf-8 -*-
import os
import json

import pandas as pd
import io
import distutils.dir_util
from collections import Counter
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

#FastText로 추가된거
from gensim.models import FastText as FT_gensim
import numpy as np

import re
#날짜 변경
from datetime import datetime, timedelta


class PlaylistEmbedding:
    # java의 생성자 같은 존재 __init__
    def __init__(self, FILE_PATH):
        self.FILE_PATH = FILE_PATH

        # word2vec의 요소들
        # 최소 1번 이상 연관이 있어야 학습한다.
        self.min_count = 2
        # 의미를 담을 벡터를 150차원으로 만든다.
        self.size = 150
        # 중심단어 기준으로 앞뒤로 210개 범위까지 학습시킨다.
        self.window = 210
        # sg = 1이면 skip-gram 아니면 CBOW
        self.sg = 1

        # 키 + 벡터를 저장함
        # KeyedVectors는 추가 교육을 지원하지 않는 대신 더 작고 RAM을 덜 차지한다.
        self.p2v_model = WordEmbeddingsKeyedVectors(self.size)

        # 유니코드 한글 시작: 44032, 끝:55199
        self.BASE_CODE, self.CHOSUNG, self.JUNGSUNG = 44032, 588, 28

        # 초성 리스트0~18
        self.CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        # 중성 리스트 0~20
        self.JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

        # 종성 리스트 0~27
        self.JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        # 여기에 자모분리된 train의 플레이리스트 제목들이 담긴다.
        self.title_list_detach = []

        # FILE_PATH가 가리키는 곳에 train, test와 most_popular_res.json, song_meta.json이 있어야 합니다.
        with open(FILE_PATH+'/train.json', encoding="utf-8") as f:
            self.train = json.load(f)
        with open(FILE_PATH+'/test.json', encoding="utf-8") as f:
            self.val = json.load(f)
        with open(FILE_PATH+'/most_popular_res.json', encoding="utf-8") as f:
            self.most_results = json.load(f)
        # song_meta 데이터를 가져온다.
        with open(FILE_PATH+'/song_meta.json', encoding="utf-8") as f:
            self.song_meta = json.load(f)

    def write_json(data, fname):
        def _conv(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            raise TypeError

        parent = os.path.dirname(fname)
        distutils.dir_util.mkpath("./arena_data/" + parent)
        with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
            json_str = json.dumps(data, ensure_ascii=False, default=_conv)
            f.write(json_str)

    def remove_seen(seen, l):
        seen = set(seen)
        return [x for x in l if not (x in seen)]

    # train, val의 곡과 태그를 플레이리스트 id를 key값으로 가지는 딕셔너리에 저장
    def get_dic(self, train, val):
        song_dic = {}
        tag_dic = {}
        data = train + val
        for q in tqdm(data):
            song_dic[str(q['id'])] = q['songs']
            tag_dic[str(q['id'])] = q['tags']
        self.song_dic = song_dic
        self.tag_dic = tag_dic

        # 여기서 토탈로 train의 곡과 태그만 보내기 때문에 모델에는 train만 학습됨
        total = list(map(lambda x: list(map(str, x['songs'])) + list(x['tags']), data))
        total = [x for x in total if len(x) > 1]
        self.total = total

    def get_w2v(self, total, min_count, size, window, sg):
        try:
            print("get_w2v 실행")
            if not (os.path.isfile("./w2v_model.model")):
                print("get_w2v 모델 학습 시작")
                # window가 210인 이유는 태그 10개와 곡 200개 꽉차있는 플레이리스트도 존재하기 때문이다.
                w2v_model = Word2Vec(total, min_count=min_count, size=size, window=window, sg=sg, iter=25)
                print("get_w2v 모델 학습 완료")
                self.w2v_model = w2v_model
                w2v_model.save("w2v_model.model")
            print("w2v_model 모델 로드")
            self.w2v_model = Word2Vec.load("./w2v_model.model")
        except OSError as e:
            print("failed to create directory!")
            raise

    def update_p2v(self, train, val, w2v_model):
        ID = []
        vec = []
        # val에 있는 곡이나 태그들 중 train에는 없어서 예외처리되는 것을 확인하기 위한 카운트
        # 이 부분은 나중에 제거해도 상관 없음
        self.yes_songs_count = 0
        self.yes_tags_count = 0
        self.no_songs_count = 0
        self.no_tags_count = 0
        for q in tqdm(train + val):
            tmp_vec = 0
            songs_vec = 0
            tags_vec = 0
            if len(q['songs']) >= 1 or len(q['tags']) >= 1:
                for x in q['songs']:
                    try:
                        songs_vec += w2v_model.wv.get_vector(str(x))
                        self.yes_songs_count += 1
                    except:
                        self.no_songs_count += 1
                for y in q['tags']:
                    try:
                        tags_vec += w2v_model.wv.get_vector(str(y))
                        self.yes_tags_count += 1
                    except:
                        self.no_tags_count += 1
                tmp_vec = songs_vec + tags_vec
            if type(tmp_vec) != int:
                ID.append(str(q['id']))
                vec.append(tmp_vec)
        # train, val의 플레이리스트 id에 해당하는 vector값을 구함
        self.p2v_model.add(ID, vec)

        # FastText

    def get_title(self, train):
        title_list = []
        for q in train:
            title_list.append(q['plylst_title'])
        self.title_list = title_list

    def jamo_str(self, text, BASE_CODE, CHOSUNG, JUNGSUNG, CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST):
        def clean_str(text):
            pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'  # E-mail제거
            text = re.sub(pattern=pattern, repl='', string=text)
            pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
            text = re.sub(pattern=pattern, repl='', string=text)
            pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
            text = re.sub(pattern=pattern, repl=' ', string=text)
            pattern = '<[^>]*>'  # HTML 태그 제거
            text = re.sub(pattern=pattern, repl=' ', string=text)
            pattern = '[^\w\s]'  # 특수기호제거
            text = re.sub(pattern=pattern, repl=' ', string=text)
            return text

        string = text
        string = clean_str(string)
        # print(string)
        sp_list = list(string)
        # print(sp_list)

        result = []
        for keyword in sp_list:
            # 한글 여부 check 후 분리
            if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:

                if keyword == ' ':
                    result.append(' ')

                if keyword in CHOSUNG_LIST or keyword in JUNGSUNG_LIST or keyword in JONGSUNG_LIST:
                    result.append('')

                else:
                    # 초성 ord->문자의 코드값을 구한다
                    char_code = ord(keyword) - BASE_CODE
                    char1 = int(char_code / CHOSUNG)
                    result.append(CHOSUNG_LIST[char1])

                    # 중성
                    char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
                    result.append(JUNGSUNG_LIST[char2])

                    # 종성
                    char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
                    if char3 == 0:
                        result.append('-')
                    result.append(JONGSUNG_LIST[char3])
            else:
                result.append(keyword)
        results_all = []
        results_all = ("".join(result))
        self.results_all = results_all

    def get_title_list(self, results_all):
        # print("".join(result)) #자모 분리 결과 출력
        title_list_detach = []
        title_list_detach.append(results_all)
        self.title_list_detach.append(title_list_detach)

    def make_title_model(self, title_list_detach):
        try:
            print("make_title_model 실행")
            if not (os.path.isfile("./FT_title_model.model")):
                print("make_title_model 모델 학습 시작")
                FT_title_model = FT_gensim(title_list_detach, size=300, window=100, min_count=1, sg=1, iter=2000)
                print("make_title_model2 모델 학습 완료")
                self.FT_title_model = FT_title_model
                FT_title_model.save("FT_title_model.model")
            self.FT_title_model = FT_gensim.load("./FT_title_model.model")
            print("make_title_model 모델 로드됨")
        except OSError as e:
            print("failed to create directory!")
            raise

    # Fasttext끝

    def get_result(self, p2v_model, song_dic, tag_dic, most_results, val, train, FT_title_model, song_meta):
        title_sentence_train = []
        for x in train:
            self.jamo_str(x['plylst_title'], self.BASE_CODE, self.CHOSUNG, self.JUNGSUNG, self.CHOSUNG_LIST,
                          self.JUNGSUNG_LIST, self.JONGSUNG_LIST)
            title_sentence_train.append(self.results_all)
        answers = []
        # 제대로 진행되고 있는지 알기 위해 세는 카운트
        # most_id는 제대로 뽑히고 있는가?
        self.most_id = []
        # ply_embedding 추천이 제대로된 플레이리스트는 몇개인가
        self.p2v_count = 0
        # 예외처리된 플레이리스트는 몇개인가
        self.except_count = 0
        # 어디서 끊기는지 정확히 알고 싶으면 옮기면서 카운트해보는 변수
        self.when_stop = 0

        # 문제유형별로 몇개의 플레이리스트가 있는 세는 카운트
        self.TNSN = 0
        self.TYSN = 0
        self.TNSY = 0
        self.TYSY = 0

        # 곡이나 태그가 100, 10개 안채워졌을 때 채우는 카운트
        self.update_song_count = 0
        self.update_tag_count = 0

        for n, q in tqdm(enumerate(val), total=len(val)):
            # 제목, 곡, 태그 유무 파악 및 개수 세기
            songs = q['songs']
            tags = q['tags']
            songs_count = len(songs)
            tags_count = len(tags)
            try:
                # 플레이리스트 임베딩하는 알고리즘(곡으로 곡추천할 때 씀)
                def ply_em(q):
                    most_id = [x[0] for x in p2v_model.most_similar(str(q['id']), topn=15)]
                    # most_vec = [x[1] for x in p2v_model.most_similar(str(q['id']), topn=15)]

                    # 원본
                    get_song = []
                    get_tag = []

                    for ID in most_id:
                        get_song += song_dic[ID]
                        get_tag += tag_dic[ID]

                    count = {}
                    for i in get_song:
                        try:
                            count[i] += 1
                        except:
                            count[i] = 1
                    count = sorted(count.items(), key=lambda x: x[1], reverse=True)

                    count2 = {}
                    for i in get_tag:
                        try:
                            count2[i] += 1
                        except:
                            count2[i] = 1
                    count2 = sorted(count2.items(), key=lambda x: x[1], reverse=True)

                    self.when_stop += 1

                    real_get_song = []
                    real_get_tag = []

                    for song in count:
                        real_get_song.append(song[0])

                    for tag in count2:
                        real_get_tag.append(tag[0])

                    # get_song = list(pd.value_counts(get_song)[:500].index)
                    # get_tag = list(pd.value_counts(get_tag)[:20].index)

                    def to_integer(dt_time):
                        return 10000 * dt_time.year + 100 * dt_time.month + dt_time.day

                    utc_time = datetime.strptime(q['updt_date'][:26], '%Y-%m-%d %H:%M:%S.%f')
                    updt = int(to_integer(utc_time))
                    true_get_song = []
                    for song_id in real_get_song:
                        issue = int(song_meta[song_id]['issue_date'])
                        if updt - issue >= 0:
                            true_get_song.append(song_id)
                        else:
                            pass

                    answers.append({
                        "id": q["id"],
                        "songs": self.remove_seen(q["songs"], true_get_song)[:100],
                        "tags": self.remove_seen(q["tags"], real_get_tag)[:10],
                    })
                    # 여기까지 오면 카운트 추가
                    self.p2v_count += 1

                    # FastText 알고리즘

                def fasttext_title(q):

                    train_ids = []
                    get_song = []
                    get_tag = []

                    self.jamo_str(q['plylst_title'], self.BASE_CODE, self.CHOSUNG, self.JUNGSUNG, self.CHOSUNG_LIST,
                                  self.JUNGSUNG_LIST, self.JONGSUNG_LIST)
                    title = self.results_all

                    F_list = FT_title_model.wv.most_similar(title, topn=60)
                    for x in F_list:
                        number = title_sentence_train.index(x[0])
                        train_ids.append(train[number]['id'])

                    for ids in train_ids:
                        get_song += song_dic[str(ids)]
                        get_tag += tag_dic[str(ids)]

                    count = {}
                    for i in get_song:
                        try:
                            count[i] += 1
                        except:
                            count[i] = 1
                    count = sorted(count.items(), key=lambda x: x[1], reverse=True)

                    count2 = {}
                    for i in get_tag:
                        try:
                            count2[i] += 1
                        except:
                            count2[i] = 1
                    count2 = sorted(count2.items(), key=lambda x: x[1], reverse=True)

                    real_get_song = []
                    real_get_tag = []

                    for song in count:
                        real_get_song.append(song[0])

                    for tag in count2:
                        real_get_tag.append(tag[0])

                    # get_song = list(pd.value_counts(real_get_song)[:200].index)
                    # get_tag = list(pd.value_counts(real_get_tag)[:20].index)

                    def to_integer(dt_time):
                        return 10000 * dt_time.year + 100 * dt_time.month + dt_time.day

                    utc_time = datetime.strptime(q['updt_date'][:26], '%Y-%m-%d %H:%M:%S.%f')
                    updt = int(to_integer(utc_time))
                    true_get_song = []
                    for song_id in real_get_song:
                        issue = int(song_meta[song_id]['issue_date'])
                        if updt - issue >= 0:
                            true_get_song.append(song_id)
                        else:
                            pass

                    answers.append({
                        "id": q["id"],
                        "songs": self.remove_seen(q["songs"], true_get_song)[:100],
                        "tags": self.remove_seen(q["tags"], real_get_tag)[:10],
                    })

                # 태그 X 곡 X 제목 O
                if tags_count == 0 and songs_count == 0:
                    self.TNSN += 1
                    fasttext_title(q)

                # 태그 O 곡 X 제목 X
                elif tags_count > 0 and songs_count == 0:
                    self.TYSN += 1
                    fasttext_title(q)

                # 태그 x 곡 O
                elif tags_count == 0 and songs_count > 0:
                    self.TNSY += 1
                    ply_em(q)

                # 태그 O 곡 O
                elif tags_count > 0 and songs_count > 0:
                    self.TYSY += 1
                    ply_em(q)

            except:
                # 예외처리되면 카운터 추가
                self.except_count += 1
                answers.append({
                    "id": q["id"],
                    "songs": most_results[n]['songs'],
                    "tags": most_results[n]["tags"],
                })

        # check and update answer
        for n, q in enumerate(answers):
            if len(q['songs']) != 100:
                answers[n]['songs'] += self.remove_seen(q['songs'], self.most_results[n]['songs'])[:100 - len(q['songs'])]
                self.update_song_count += 1
            if len(q['tags']) != 10:
                answers[n]['tags'] += self.remove_seen(q['tags'], self.most_results[n]['tags'])[:10 - len(q['tags'])]
                self.update_tag_count += 1
        self.answers = answers

    def run(self):
        # Word2Vec ply_embedding
        self.get_dic(self.train, self.val)
        self.get_w2v(self.total, self.min_count, self.size, self.window, self.sg)
        self.update_p2v(self.train, self.val, self.w2v_model)

        # FastText ply_title
        self.get_title(self.train)
        for string in self.title_list:
            self.jamo_str(string, self.BASE_CODE, self.CHOSUNG, self.JUNGSUNG, self.CHOSUNG_LIST, self.JUNGSUNG_LIST,
                          self.JONGSUNG_LIST)
            self.get_title_list(self.results_all)
        self.make_title_model(self.title_list_detach)

        # 곡과 태그 채우는 함수
        self.get_result(self.p2v_model, self.song_dic, self.tag_dic, self.most_results, self.val, self.train,
                        self.FT_title_model, self.song_meta)

        self.write_json(self.answers, 'results.json')
        print("results 작성 완료")

    def train_model(self):
        # Word2Vec ply_embedding
        self.get_dic(self.train, self.val)
        self.get_w2v(self.total, self.min_count, self.size, self.window, self.sg)
        self.update_p2v(self.train, self.val, self.w2v_model)

        # FastText ply_title
        self.get_title(self.train)
        for string in self.title_list:
            self.jamo_str(string, self.BASE_CODE, self.CHOSUNG, self.JUNGSUNG, self.CHOSUNG_LIST, self.JUNGSUNG_LIST,
                          self.JONGSUNG_LIST)
            self.get_title_list(self.results_all)
        self.make_title_model(self.title_list_detach)
