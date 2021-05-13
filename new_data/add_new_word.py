from numpy.random import RandomState
import pandas as pd
import numpy as np
df = pd.read_csv('ICHI-dataset/data/ichi/train.tsv',sep='\t')
#df=df[:4]
med = pd.read_excel('patient-friendly_term_list_v24.0.xlsx', engine='openpyxl')
med_concept=med['LLT']
import re
def clean_text(text):
    #text = text.lower()
    text = re.sub(r"\!", "", text)
    text = re.sub(r"\.", "", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

threshould=85

from fuzzywuzzy import fuzz
from numpy import nan

def check_sim(txt):
  max_sim=0
  s="-1"
  for med_word in med_concept:
    sim=fuzz.ratio(med_word,txt)
    #sim_per=fuzz.partial_ratio(med_word,txt)
    if sim>max_sim:
      max_sim=sim
      s=med_word
    #if sim_per>max_sim:
    #  max_sim=sim_per
    #  s=med_word
    
  return (max_sim,s)

  #df_extra_sentence

extra_med_word=[]
extra_mid_word_with_sim=[]
k=0
for sen in df['Question']:
  txt_word=clean_text(sen).split(' ')
  set1=set()
  for i in range(len(txt_word)):
    s=txt_word[i]
    (sim,s1)=check_sim(s)
  
    if sim>threshould:
      set1.add(s)
      extra_mid_word_with_sim.append((s,sim,s1))
      continue
    elif i+1<len(txt_word):
      s=s+' '+txt_word[i+1]
      (sim,s1)=check_sim(s)
      if sim>threshould:
        set1.add(s)
        extra_mid_word_with_sim.append((s,sim,s1))
        i=i+1
        continue
    elif i+2<len(txt_word):
      s=s+' '+txt_word[i+2]
      (sim,s1)=check_sim(s)
      if sim>threshould:
        set1.add(s)
        extra_mid_word_with_sim.append((s,sim,s1))
        i=i+2
        continue
  
 
  
  print("id=",k,set1)
  k=k+1
  if len(set1)>0:
    extra_med_word.append("|".join(list(set1)))
  else:
    extra_med_word.append(np.nan)
  
  
df_temp=df

tmp_df_concepts = pd.DataFrame(list(extra_med_word),columns=["Concepts"])

for i in range(len(df_temp)):
  set1=set()
  
  if df_temp["Concepts"].isnull().iloc[i]==False:
    set1.update(df_temp["Concepts"].iloc[i].split('|'))
  if tmp_df_concepts["Concepts"].isnull().iloc[i]==False:
    set1.update(tmp_df_concepts["Concepts"].iloc[i].split('|'))
  print(i,set1)
  if(len(set1)>0):
    df_temp.iloc[i][3]="|".join(list(set1))
  
df_temp.to_csv('final_train_result.tsv', sep="\t",index=False)
df_extra_med_word=pd.DataFrame(extra_mid_word_with_sim,columns=["word_in_doc","similarity","actual_medical_word"])
df_extra_med_word.to_csv("df_extra_med_word_train.csv",index=False)
