#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


start = time.time()


# In[7]:


text_data = pd.read_csv('Corona_NLP_train2.csv')


# In[4]:


text_data.head()


# In[5]:


text_data.info()


# ## step one unique 

# In[6]:


text_data['Sentiment'].unique()


# In[7]:


text_data['Sentiment'].value_counts()


# In[8]:


text_data['Sentiment'].value_counts().idxmax()


# ### subset data frame
# ### filter 
# ### groupby or pivot table
# ### step 2 

# In[9]:


Sentiments = text_data["Sentiment"]
Sentiments.head()


# In[10]:


Sentiments_date = text_data[["Sentiment","TweetAt"]]
Sentiments_date.head()


# In[11]:


Sentiments_postive = Sentiments_date [Sentiments_date ["Sentiment"] == "Extremely Positive"]
Sentiments_postive.head()


# In[ ]:





# In[12]:


group = Sentiments_postive.groupby('TweetAt')['Sentiment'].count()

group.sort_values(ascending=False).head()


# In[13]:


# flights_by_carrier = Sentiments_postive.pivot_table(index='TweetAt', columns='Sentiment', aggfunc='count')
# flights_by_carrier.head()


# In[14]:


# flight_delays_by_day = Sentiments_postive.pivot_table(index='TweetAt', values='Sentiment', aggfunc='count')
# flight_delays_by_day


# ## step 3  convert the messages to lower case, replace non-alphabetical characters with whitespaces and ensure that the words of a message are separated by a single whitespace

# In[15]:


# df.apply(lambda x: x.astype(str).str.upper())


# In[16]:


# messages = text_data["OriginalTweet"].str.lower()
# messages.head()


# In[17]:


text_data['OriginalTweet'].loc[0]


# In[18]:


text_data_new = text_data.copy()
text_data_new.head()


# In[19]:


# text_data_new['OriginalTweet'] = text_data_new['OriginalTweet'].str.replace('http\S+|www.\S+', '', case=False)


# In[20]:


text_data_new['OriginalTweet'].loc[0]


# ### here we have converted to lower case and removed noise such as non-alphabetical characters with whitespaces and ensure that the words of a message are separated by a single whitespac

# In[21]:



text_data_new['OriginalTweet'] = text_data_new['OriginalTweet'].str.replace('http\S+|www.\S+', '', case=False)
text_data_new['OriginalTweet'] = text_data_new['OriginalTweet'].str.replace('[^0-9a-zA-Z]+', ' ', case=False)
text_data_new['OriginalTweet'] = text_data_new["OriginalTweet"].str.lower()


# In[22]:


text_data_new['OriginalTweet'].loc[0]


# ### Tokenize the tweets (i.e. convert each into a list of words), count the total number
# ### of all words (including repetitions), the number of all distinct words and the 10 most frequent words in the #### corpus

# In[30]:


from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()


# In[31]:


tweet_tokens = []
for sent in text_data_new:
    print(tweet_tokenizer.tokenize(sent))
    tweet_tokens.append(tweet_tokenizer.tokenize(sent))


# In[32]:


text_data_new


# In[33]:


text_data_new['OriginalTweet']


# In[34]:


from sklearn.feature_extraction.text import CountVectorizer

df_list = text_data_new['OriginalTweet'].values.tolist()
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df_list)
X_train_counts


# In[35]:


print('Total unique words : ',len(count_vect.get_feature_names()))


# In[36]:


token_count_df = pd.DataFrame(X_train_counts.toarray(),columns = count_vect.get_feature_names())


# In[37]:


print('Total word with repitions : ',token_count_df.sum().sum())


# In[38]:


vocabulary_sort_alldocs  = token_count_df.sum().sort_values(ascending= False)


# In[39]:


vocabulary_sort_alldocs.head(10)


# In[40]:


count_vect_sw = CountVectorizer(stop_words='english', min_df=2)
X_train_counts_sw = count_vect_sw.fit_transform(df_list)
print('Total unique words : ',len(count_vect_sw.get_feature_names()))


# In[41]:


X_train_counts_sw


# ### Remove stop words, words with ≤ 2 characters and recalculate the number
# ### of all words (including repetitions) and the 10 most frequent words in the modified corpus

# In[42]:


token_count_df_sw = pd.DataFrame(X_train_counts_sw.toarray(),columns = count_vect_sw.get_feature_names())


# In[43]:


print('Total word with repitions : ',token_count_df_sw.sum().sum())


# In[44]:


vocabulary_sort_alldocs_sw  = token_count_df_sw.sum().sort_values(ascending= False)


# In[45]:


print(vocabulary_sort_alldocs_sw.head(10))


# #### Histogram

# In[46]:


num_docs = token_count_df_sw.shape[0]
token_doc_occ_freq = (token_count_df_sw != 0).sum(0) / num_docs
token_doc_occ_freq = token_doc_occ_freq.sort_values(ascending= False)

# Convert words to numbers
token_doc_occ_freq_num = token_doc_occ_freq.reset_index(drop=True)

token_doc_occ_freq_num.sort_values(ascending= True).reset_index(drop=True) .plot(title = 'Word Frequency (Increasing Order)')
plt.figure()
token_doc_occ_freq_num.plot(title = 'Word Frequency')
plt.figure()
token_doc_occ_freq_num.plot(title = 'Word Frequency (log scale)',logx=True, logy = True)


# In[47]:


vocabulary_sort_alldocs_sw.head(10)


# In[48]:


y = text_data_new['Sentiment']


# In[49]:


model = MultinomialNB()
model.fit(X_train_counts_sw,y)
y_pred = model.predict(X_train_counts_sw)


# In[50]:



clf_report_dict = classification_report(y, y_pred, output_dict=True)
pd.DataFrame(clf_report_dict).round(2).T


# In[8]:


end = time.time()
print(f"Runtime for the program is {end-start}")


# In[ ]:





# In[ ]:




