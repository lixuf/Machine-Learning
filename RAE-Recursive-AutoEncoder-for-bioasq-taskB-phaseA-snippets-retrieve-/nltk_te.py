import nltk
from nltk.corpus import stopwords
print(stopwords)
from nltk.tokenize import word_tokenize,sent_tokenize
sentence="Is there evidence that tomato juice lowers cholesterol levels?"
tokens = nltk.word_tokenize(sentence)
print(tokens)


#词干提取
from nltk.stem import PorterStemmer
ps = PorterStemmer()
for w in tokens:
    print(ps.stem(w))

#提取名词短语
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

tokenized = nltk.pos_tag(tokens)
grammar = r"""
    NP: {<DT|PRP\$>?<JJ>*<NN>}
        {<NNP>+}
"""
chunkParser = nltk.RegexpParser(grammar)
chunked = chunkParser.parse(tokenized)
for i in chunked:
    try:
        if i.label()=="NP":
            leaf_values = i.leaves()
            for e in leaf_values:
                print(e[0])
    except:
        print(i)
