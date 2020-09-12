#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import collections
import math
import pymorphy2

class Jobsclassifier:
    
    def __init__(self):
        
        self.morph = pymorphy2.MorphAnalyzer()
    
    def lemmatize(self, text):
        
        words = text.split() 
        
        res = list()
        
        for word in words:
            p = self.morph.parse(word)[0]
            res.append(p.normal_form)

        return res
    
    def description_processing(self, line):
        
        a = ['.', '-', '(', ')', ',', ':', ';']
        
        for i in range(0, len(a)):
        
            line = line.replace(a[i], ' ')
            
        line = line.split()
    
        return line
    
    def calculation_tf(self, text):
        
        tf_text = collections.Counter(text)
    
        for i in tf_text:
        
            tf_text[i] = tf_text[i]/float(len(text))
        
        return tf_text

    def calculation_idf(self, word, array):
        
        n = 0
        
        for i in array:
            
            if word in i:
                
                n += 1
                
        return math.log10(len(array)/n)
    
    def calculation_tfidf(self, array):
        
        documents_list = []
        
        for line in array:

            tf_idf_dictionary = {}

            calculated_tf = self.calculation_tf(self.description_processing(line))

            for word in calculated_tf:
            
                if (calculated_tf[word] * self.calculation_idf(word, array)) > 0.03:
                
                    tf_idf_dictionary[word] = calculated_tf[word] * self.calculation_idf(word, array)
            
            documents_list.append(tf_idf_dictionary)

        return documents_list
    
    def fit(self, X, y, category):
        
        big = self.calculation_tfidf(X)

        best = {}
        
        self.category = category
        
        for i in range(0, len(category)):
            
            best.update({category[i]: {}})

        for i in range(0, len(big)):
            
            for key in big[i].keys():
                
                key = self.lemmatize(key)[0]
                
                if key not in best.get(y[i]):
                    
                    (best.get(y[i])).update({key: 1})
                    
                else:
                    
                    (best.get(y[i])).update({key: (best.get(y[i])).get(key) + 1})
            
        self.best_list = []

        for big_key in best.keys():
            
            a = []
            
            for small_key in best[big_key].keys():
                
                if (best[big_key]).get(small_key) >= 4:
                    
                    a.append(small_key)
                    
            if len(a) <= 20:
                
                for small_key in best[big_key].keys():
                    
                    if len(a) <= 20:
                        
                        if (best[big_key]).get(small_key) >= 2:
                            
                            a.append(small_key)
        

                         
            self.best_list.append(a)
    
    def predict(self, X):
        
        pred = []
        
        morph = pymorphy2.MorphAnalyzer()
        
        for i in range(0, len(X)):
            
            a = 0
            
            k = []
            
            for n in range(0, len(self.category)):
                
                k.append(0)
            
            for word in self.description_processing(X[i]):
        
                word = (self.morph.parse(word)[0]).normal_form
            
                for j in range(0, len(self.category)):
                    
                    if word in self.best_list[j]:
                        
                        k[j] = k[j] + 1
                        
            for j in range(0, len(self.category)):
                
                if k[j] == np.max(k) and a == 0:
                    
                    pred.append(self.category[j])
                    
                    a = 1
                
        return pred
    
    
data_train = pd.read_csv('train.csv')
data1 = data_train['name'] + ' ' + data_train['description']

data_test = pd.read_csv('test.csv')
data2 =  data_test['name'] + ' ' + data_test['description']

category = ['Менеджер', 'Искусство', 'Рабочий', 'Дизайнер', 'Специалист',
            'СМИ', 'Врач', 'other', 'Инженер', 'IT', 'Право', 'Учитель',
            'Агент']

model = Jobsclassifier()
model.fit(data1, data_train1['category'], category)

pred = model.predict(data2)

Id = []
pred = np.array(pred)
for i in range (0, 106):
    Id.append(i)
Id = np.reshape(Id, pred.shape)
Answer = pd.DataFrame(Id)
Answer.columns = ['id']
Answer['category'] = pred
Answer.to_csv("Answer.csv", index=False)

