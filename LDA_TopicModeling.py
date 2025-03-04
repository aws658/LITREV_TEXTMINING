# inspired by:
    # https://nicharuc.github.io/topic_modeling/#topic=0&lambda=1&term=  
    # (Optimizing LDA Topic Model for Interpretability)
    # https://medium.com/analytics-vidhya/topic-modeling-using-gensim-lda-in-python-48eaa2344920 
    # (Topic Modeling using Gensim-LDA in Python)
    # https://datascience.oneoffcoder.com/topic-modeling-gensim.html#Coherence-Scores
    # (section 2.6 coherence score interpretation)
    # https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
    # (various visualization that can be considered)
    # https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know
    # general read to understand/interpret pyldavis output
"""
Created on Mon Oct  9 11:16:21 2023
@author: Atiq Siddiqui - awsiddiqui@iau.edu.sa
"""

# ---------------------------------------
# key parameter setting
filename='Dataset_Gas_Transportation.xlsx'                       #data filename
# ~ ~ ~ ~ ~ ~ ~
#set minimum bigram and trigram frequenc                    ---- setting for MSBA
freq_bi=5  #use smaller values for smaller datasets       ---- 2
freq_tri=5  # use smaller values for smaller datasets     ---- 2
analysis_basis=5 # 5=combined, 1=title, 2=keywords, 3=keywords_Plus, 4=abstract
maxNgramsConsidered=150 #MA ngrams considered in analysis   --- 150

# Model parameters
iterations =16 # chose between 10 and 15 -- see if coherence curver is smooth or not ---- 10
chunksize = 110  #number of docs considered each iteration  - 60 for 100 docs ------      60-80
#                --- larger chunk and smaller iterations lead to good results
randseed=60 # Fixing seed is to reproduce results - use one which increase coherence ----- 60 or 80
passes=50 # times the model is updated
# Try seed that gives to nearest to zero optimal coherence

# Parameters used in optimization of model
k_start=2 # minimum number of topics considered - test iteratively            ----- 5
k_max=15 # minimum number of topics considered                                ----- 25
max_words_considered_in_topic_representation=6 # max considered in topics labelling

# ---------------------------------------

# to stop certain warning poping-up
import warnings
warnings.filterwarnings('ignore') 
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ------------------
# necessary package imports
# ------------------
# print ("importing necessary libraries...")
import spacy
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.corpus import wordnet
wordnet=wordnet.words()
# nltk.download('stopwords')  -- incase of error of Resource stopwords not found.
# nltk.download('wordnet')    -- incase of error of Resource wordnet not found.
# nltk.download('averaged_perceptron_tagger')    -- nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

import re
import string
import pandas as pd
import numpy as np
import xlsxwriter
#from stop_word_list import *
from cleantext import *

from wordcloud import WordCloud

import gensim
from gensim import corpora
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------
# Loading the data
# ------------------
print ("loading the dataset...")
df = pd.read_excel(filename)  # filename is defined as the first parameter above
df.dropna(subset=df.columns[df.shape[1]-1], inplace=True)
df.to_string()
df.head()
headers=list(df.columns.values)
lenis=len(headers)
print ("number of records found: ",df.shape[0])


# -----------------
#Loading custom extended stopwords list
dftemp = pd.read_excel("extendedStopwords.xlsx")
stopEx = dftemp.values.tolist()
stop=stop+stopEx


# ---------------------------------------------------
# cleaning the datafile except the first two columns
# first column is paper ID and second column is 
# ---------------------------------------------------
print ("cleaning the datafile except the first two columns - ID and year")

for i in range(2,df.shape[1]):
    # clean the file using clean text for unicode fixes, to_ascii, lower etc.
    #df.iloc[:,i]=df.iloc[:,i].map(clean)  #https://pypi.org/project/clean-text/
    # removes double space with a single space
    df.iloc[:, i] = df.iloc[:, i].apply(str.lower) # coverting all to lower case
    df.iloc[:, i] = df.iloc[:, i].str.replace('  ', ' ')
    # removes question mark
    df.iloc[:, i] = df.iloc[:, i].str.replace('?', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('- ', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('(', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace(')', '')
    # removes the copy right statement where present
    df.iloc[:, i] = df.iloc[:, i].str.replace('b.v. all rights reserved', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('Elsevier Ltd. All rights reserved.', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('All rights reserved.', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('(C)', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('(c) elsevier ltd right reserved', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('<sub></sub>', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('john wiley & son ltd', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('john wiley', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('©️', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('author', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('licensee', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('mdpi', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('elsevier', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('bv', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('ltd right reserved', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('inc', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('taylor & francis', '')
    df.iloc[:, i] = df.iloc[:, i].str.replace('dump', '')
    
    
    
    
    # removes nan
    df.iloc[:, i] = df.iloc[:, i].str.replace('nan', '')
    # removes puntuation marks
    df.iloc[:,i]=[re.sub(r'\.|,|(|)|:|;|%|"|\n','', str(x)) for x in df.iloc[:,i]]
    # removes numbers from the text
    df.iloc[:,i]=[re.sub(r'\d+','', str(x)) for x in df.iloc[:,i]]
    # removes stopwords - provided in nltk.corpus stopwords list
    # print(stop) # to see the list
    df.iloc[:,i] = df.iloc[:,i].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    # adding a space at the end - needed when combining the columns
    df.iloc[:,i]=df.iloc[:,i]+" "

print(df)
# creating final column on which analysis is applied based on the anlysis_basis set
if analysis_basis==5:
    df['Combined'] = df[[headers[lenis-4],headers[lenis-3],headers[lenis-2],headers[lenis-1]]].agg(''.join, axis=1)
else:
    df['Combined'] = df.loc[:, headers[analysis_basis+1]]


# -------------------------------------------------------------------
# lemmatizing the individual and combined columns
print ("lemmatizing the data...")
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
     words = text.split()
     words = [lemmatizer.lemmatize(word) for word in words]
     return ' '.join(words)
# individual columns
for i in range (0,lenis-2):
    #print(headers[2+i])
    df[headers[2+i]] = df[headers[2+i]].apply(lemmatize_words)
# combined columns                    
df['Combined'] = df['Combined'].apply(lemmatize_words)


# ----------------------------
# saving the updated cleaned datafile.   
print ("saving the cleaned and combined data to combined.xlsx...")
df.to_excel("combined.xlsx")


#------------------
# creating bigrams
print()
print ("creating bi and trigrams...")

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_documents([comment.split() for comment in df.Combined])
# Filter only those that occur at least 50 times
finder.apply_freq_filter(freq_bi)
bigram_scores = finder.score_ngrams(bigram_measures.pmi)

bigram_pmi = pd.DataFrame(bigram_scores)
bigram_pmi.columns = ['bigram', 'pmi']
bigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

#------------------
# creating trigrams
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = nltk.collocations.TrigramCollocationFinder.from_documents([comment.split() for comment in df.Combined])
# Filter only those that occur at least 50 times
finder.apply_freq_filter(freq_tri)
trigram_scores = finder.score_ngrams(trigram_measures.pmi)
trigram_pmi = pd.DataFrame(trigram_scores)
trigram_pmi.columns = ['trigram', 'pmi']
trigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

print ("birgam with pmi scores")
print (bigram_pmi)
print()
print ("birgam with pmi scores")
print (bigram_pmi)
print()
#------------------
#------------------
# Filtering bi and tri grams for stopwords
print()
print ("filtering bi and trigrams for sotp words...")
#------------------
# Filter for bigrams with only noun-type structures
def bigram_filter(bigram):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['NN']:
        return False
    if bigram[0] in stop or bigram[1] in stop:
        return False
    if 'n' in bigram or 't' in bigram:
        return False
    if 'PRON' in bigram:
        return False
    return True

# Filter for trigrams with only noun-type structures
def trigram_filter(trigram):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ','NN']:
        return False
    if trigram[0] in stop or trigram[-1] in stop or trigram[1] in stop:
        return False
    if 'n' in trigram or 't' in trigram:
         return False
    if 'PRON' in trigram:
        return False
    return True

# Can set pmi threshold to whatever makes sense - eyeball through and select threshold where n-grams stop making sense
# choose top -maxNgramsConsidered- ngrams in this case ranked by PMI that have noun like structures
filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram:\
                                              bigram_filter(bigram['bigram'])\
                                              and bigram.pmi > 5, axis = 1)][:maxNgramsConsidered]

filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: \
                                                 trigram_filter(trigram['trigram'])\
                                                 and trigram.pmi > 5, axis = 1)][:maxNgramsConsidered]


bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
#bigrams=[x.replace(" ","_") for x in bigrams]
trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]
#trigrams=[x.replace(" ","_") for x in trigrams]

print ("filtered birgam")
print (bigrams)
print()
print ("filtered trirgam")
print (bigram_pmi)
print()


# ---------------------------------------
# Concatenate n-grams
def replace_ngram(x):
    for gram in trigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    return x

reviews_w_ngrams = df['Combined'].copy()
reviews_w_ngrams = reviews_w_ngrams.to_frame()
df.Combined = reviews_w_ngrams.Combined.map(lambda x: replace_ngram(x))
#df.Combined = reviews_w_ngrams.Combined.map(lambda x: bigrams)
#df.Combined = reviews_w_ngrams.Combined.map(lambda x: trigrams)

reviews_w_ngrams = reviews_w_ngrams.Combined.map(lambda x: [word for word in x.split()\
                                                 if word not in stop\
                                                              and word not in wordnet\
                                                              and len(word) > 2])


# ----------------------------
# saving the updated cleaned datafile.   

    
    
# ---------------------------------------    
# Filter for only nouns or noun/verds
def noun_only(x):
    pos_comment = nltk.pos_tag(x)
    # to filter both noun and verbs
    filtered = [word[0] for word in pos_comment if word[1] in ['NN']]
    # to filter both noun and verbs
    #filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered
final_reviews = reviews_w_ngrams.map(noun_only)
#final_reviews.to_excel("grams.xlsx")




# --------------------------------------
# Generate a word cloud
print("~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~")
print ("Generating Word cloud....")
data_processed=reviews_w_ngrams.to_string()

wordcloud = WordCloud(width=800, height=800, background_color='white',
                      min_font_size=5).generate(data_processed)

# Display the word cloud using matplotlib
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
print("~~~~~~~~~~~~~~")
print("")
# --------------------------------------




# --------------------------------------
# Topics Model
print("~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~")
print ("creating the model....")
# ----------------------------------
# ------  Create DICTIONARY --------
# ----------------------------------
dictionary = corpora.Dictionary(final_reviews)
# ----------------------------------
# ---- Creating CORPUS: Term Document Frequency -----
# ----------------------------------
doc_term_matrix = [dictionary.doc2bow(doc) for doc in final_reviews]
# ----------------------------------


# ----------------------------------
# Creating and Optimizing the Model
# ----------------------------------
print ("optimizing the model i.e., finding the number of topics with max coherence....")
coherence = []    
for k in range(k_start,k_max):
    print('Round: '+str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=k, random_state=randseed, id2word = dictionary, passes=passes,\
                   iterations=iterations, chunksize = chunksize, eval_every = None, update_every=1)
    cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=final_reviews, dictionary=dictionary, coherence='u_mass')
    coherence.append((k,cm.get_coherence()))
        

# ploting the coherence vs k results            
x_val = [x[0] for x in coherence]
y_val = [x[1] for x in coherence]
plt.figure(figsize=(6,3))
plt.plot(x_val,y_val)
plt.scatter(x_val,y_val)
plt.title('Number of Topics vs. Coherence')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence')
plt.xticks(x_val)
plt.show()


#----------------- 
#determining optimal topics or k from the above data
first,second=zip(*coherence)
opt_k=second.index(max(list(second)))+k_start
#opt_k=11
print ("optimal k: ",opt_k)
# --------------------
# rerunning the model optimal k and saving the topic data
ldamodel = Lda(doc_term_matrix, num_topics=opt_k, random_state=randseed, id2word = dictionary, passes=40,\
               iterations=iterations, chunksize = chunksize, eval_every = None, update_every=1)
topics=ldamodel.show_topics(opt_k, num_words=max_words_considered_in_topic_representation, formatted=False)

#print(topics)


# saving topic data in df and excel
nameis_ls=[]
topicis_ls=[]
for i in range (opt_k):
    nameis_ls.append("Topic"+str(i+1))
    topicis=""
    for j in range(len(topics[i][1])):
        topicis=topicis+topics[i][1][j][0]+", "
    topicis_ls.append(topicis)
    
topic_data={"TopicID":nameis_ls, "Topics": topicis_ls}
df_topics=pd.DataFrame(topic_data)     
print("the topics found are:")
print(df_topics)
# exporting to excel
writer = pd.ExcelWriter("Topics.xlsx", engine = 'xlsxwriter')
df_topics.to_excel(writer,sheet_name ="Topics")

# Recording topics to document relationship data
doc = ldamodel[doc_term_matrix]
l=[ldamodel.get_document_topics(item) for item in doc_term_matrix]



#------------------------------------------
# process the results for topic analysis
print ("processing the results")
topic_data =  pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False, mds='mmds')
#graphical output of topics analysis
pyLDAvis.save_html(topic_data, 'topics_viz.html')
print ("Results saved in topics_viz.html")


#------------------------------------------
# preparing data for clustered heatmap
nameis_ls.insert(0,"Doc")
df_dataHM=pd.DataFrame(columns=nameis_ls)
for i in (range(len(df))):
    lst_temp = [0] * opt_k
    lst_temp.insert(0,i)
    for j in range(len(l[i])):
        lst_temp[l[i][j][0]+1]=l[i][j][1]
    lst_temp[0]=df.iloc[i][0]
    df_dataHM.loc[i]=lst_temp
print(df_dataHM)
df_dataHM.to_excel(writer,sheet_name ="Doc_to_Topic")
extracted_col = df["Publication Year"]
writer.close()

#------------------------------------------
# Display the heatmap
# NOTE: WHILE IT IS PROVIDING A CLUSTERED HEATMAP ---
# USE THE ONE PROVIDED VIA ORANGE FILE Generate_Final_Heatmap
df_dataHM.pivot(index='Doc', columns=list(df_dataHM.columns.values)[1:])
Doc = df_dataHM.pop("Doc")
y_axis_labels = Doc
sns.set_theme(rc={'figure.dpi': 250}, font_scale=0.65)
g=sns.clustermap(df_dataHM, cmap="YlOrBr", yticklabels=y_axis_labels)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 4)
plt.show()
#------------------------------------------




import csv

with open(..., 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(bigrams)