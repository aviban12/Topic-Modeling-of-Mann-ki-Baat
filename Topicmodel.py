import gensim
from bokeh.layouts import gridplot, row, column
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer, PorterStemmer, text
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from bokeh.plotting import output_file , figure ,show
sentence= []
rm_stop = []
li = []
stop = ['fellow','countrymen', 'dear', 'greetings','new','feel','dirt','need','days','years','baat','world','babasaheb', 'ambedkar', 'mann','country','india','world' 'namaskar', 'feeling', 'like', 'today','day','year']
wordnet_lemmatizer = WordNetLemmatizer()
output_file('plot.html')
layout = gridplot()
def load():
    df = pd.read_csv("mann_ki_baat.csv")
    for i in range(0, len(df)):
        topic(df, i)
def topic(df1,k):
    print(k)
    p = figure(title="Graphical representation of Topics", x_axis_label='Topic', y_axis_label='Value',plot_width=400 , plot_height = 400)
    valuelist = []
    strlist = []
    tokens = []
    lemmatized = []
    c = 0
    speech = df1['Speech_text'][k]
    blob = TextBlob(speech)
    noun =  blob.noun_phrases
    stemmer = PorterStemmer()
    sent= [w for w in sent_tokenize(speech.lower())]
    for i in range(0, len(sent)):
        tokens.append([w for w in word_tokenize(sent[i].lower()) if w.isalpha() and w not in STOPWORDS and not len(w)<=2 and w not in stop and w not in noun])
    for i in tokens:
        lemm = [wordnet_lemmatizer.lemmatize(t) for t in i]
        lemmatized.append(lemm)
    dictionary = Dictionary(lemmatized)
    corpus = [dictionary.doc2bow(doc) for doc in lemmatized]
    '''tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus[0]]
    sorted_tfidf = sorted(corpus_tfidf, key=lambda w: w[1], reverse=True)
    for term_id, weight in sorted_tfidf:
        strlist.append(dictionary.get(term_id))
        valuelist.append(weight)
    #plt.barh(strlist, valuelist)
    #plt.show()'''
    t = 0
    lda_model = gensim.models.LdaMulticore(corpus,num_topics=10, id2word=dictionary, passes=2)
    K = lda_model.num_topics
    topicWordProbMat = sorted(lda_model.print_topics(K), key=lambda tup: tup[1], reverse=True)
    for line in topicWordProbMat:
           tp, w = line
           probs = w.split("+")
           y=0
           for pr in probs:
                print(pr)
                a = pr.split("*")
                valuelist.append(float(a[0]))
                y += 1
                strlist.append(y)
                c = c + 1
                yr = str(df1['year'][k])
                mon = str(df1['month'][k])
                plt.title(mon + " " + yr)
                plt.xlabel("Value")
                plt.ylabel("Topic")
                if(c>10):
                    break
    print(valuelist)
    print(strlist)
    #plt.barh(strlist,valuelist,color='r')
    #plt.show()
    p.h
    bar(y=valuelist,
           height=0.1,
           left=0,
           right=strlist,
           color="navy")
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    layout = row(p)
    show(layout)
    #plt.barh(strlist, valuelist, color='g',alpha=0.6)
    #plt.show()
load()
