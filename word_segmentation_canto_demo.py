# Commented out IPython magic to ensure Python compatibility.
# %pip install pycantonese

# %pip install mecab-python3
# %pip install unidic-lite

# %pip install jieba

# Ref: https://pycantonese.org/word_segmentation.html
import pycantonese
from pycantonese.word_segmentation import Segmenter

# Ref: https://github.com/SamuraiT/mecab-python3
import MeCab

# Ref: https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/nlp/chinese-word-seg.html
import jieba

# ======================================================

# Target text 
text = "兒子生性病母倍感安慰"

# ======================================================
## pycantonese

pycantonese.segment(text)

segmenterA = Segmenter(allow={"病母"})
pycantonese.segment(text, cls= segmenterA)

segmenterD = Segmenter(disallow={"性病"})
pycantonese.segment(text, cls= segmenterD)

segmenterM = Segmenter(max_word_length= 2)
pycantonese.segment(text, cls= segmenterD)

# ======================================================
## MeCab
wakati = MeCab.Tagger("-Owakati")
wakati.parse("pythonが大好きです").split()

wakati.parse(text).split()

# ======================================================
## jieba

text_jb = jieba.lcut(text)
print(' | '.join(text_jb))

# ======================================================
# Others:

# 使用Owakati進行分詞處理
# Ref: https://qiita.com/uichi/items/dd05f1d83ed22911f420

# Japanese BERT-base (MeCab + BPE) | Hugging Face
# hitachi-nlp/bert-base-japanese_mecab-bpe
# Ref: https://huggingface.co/hitachi-nlp/bert-base-japanese_mecab-bpe

