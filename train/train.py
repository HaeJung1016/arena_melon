# -*- coding: utf-8 -*-
from PlaylistEmbedding import PlaylistEmbedding

FILE_PATH = '../../dataset'
U_space = PlaylistEmbedding(FILE_PATH)
U_space.train_model()


