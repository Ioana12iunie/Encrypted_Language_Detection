#!/usr/bin/env python
# coding: utf-8

# In[1]:


#SUBMISIA FINALA 1

#deschid fisierele si extrag datele pentru antrenat, validare, testare

fisier_input_train = open("./train_samples.txt", "r", encoding="utf-8")
date_pentru_antrenat = []
for date_input in fisier_input_train.readlines():
  constructie_sample = ''
  lungime_sample = len(date_input) 
  for index in range(lungime_sample):
    if index > 6: #nu iau in considerare id ul si spatiul de dupa acesta
      constructie_sample += date_input[index]
  date_pentru_antrenat.append(constructie_sample)


#verificarea codului pentru preluarea datelor pentru antrenat

#print(len(date_pentru_antrenat)) -> 1000

# with open('samples.txt', 'w+') as f:
#   for sample in date_pentru_antrenat:
#     f.write(sample)

fisier_input_validation = open("./validation_samples.txt", "r", encoding="utf-8")
date_pentru_validare = []
for date_input in fisier_input_validation.readlines():
  constructie_sample = ''
  lungime_sample = len(date_input) 
  for index in range(lungime_sample):
    if index > 6: #nu iau in considerare id ul si spatiul de dupa acesta
      constructie_sample += date_input[index]
  date_pentru_validare.append(constructie_sample)

#verificarea codului pentru preluarea datelor pentru validare

# print(len(date_pentru_validare)) -> 5000

# with open('samples.txt', 'w+') as f:
#   for sample in date_pentru_validare:
#     f.write(sample)


fisier_input_testare = open("./test_samples.txt", "r", encoding="utf-8")
date_pentru_testare = []
ids_pentru_testare = [] #voi retine id-urile necesare pentru a construi submisia
for date_input in fisier_input_testare.readlines():
  constructie_sample = ''
  constructie_id = '' 
  lungime_sample = len(date_input) 
  for index in range(lungime_sample):
    if index > 6: #nu iau in considerare id ul si spatiul de dupa acesta
      constructie_sample += date_input[index]
    else:
      if date_input[index] != ' ': 
        constructie_id += date_input[index]
  date_pentru_testare.append(constructie_sample)
  ids_pentru_testare.append(constructie_id)


#verificarea codului pentru preluarea datelor pentru testare

# print(len(date_pentru_testare)) -> 5000

# with open('samples.txt', 'w+') as f:
#   for sample in date_pentru_testare:
#     f.write(sample)

#print(len(ids_pentru_testare)) -> 5000
# with open('samples.txt', 'w+') as f:
#   for sample in ids_pentru_testare:
#     f.write(sample)
#     f.write('\n')


#deschid fisierele si extrag etichetele pentru antrenat si validare

fisier_input_etichete_antrenat = open("./train_labels.txt", "r", encoding="utf-8")
etichete_pentru_antrenat = []
for etichete_input in fisier_input_etichete_antrenat.readlines():
  etichete_pentru_antrenat.append(etichete_input[7])

#verificarea codului pentru preluarea etichetelor pentru antrenare

# print(len(etichete_pentru_antrenat)) #-> 10000

# with open('samples.txt', 'w+') as f:
#   for sample in etichete_pentru_antrenat:
#     f.write(sample)
#     f.write('\n')

fisier_input_etichete_validare = open("./validation_labels.txt", "r", encoding="utf-8")
etichete_pentru_validare = []
for etichete_input in fisier_input_etichete_validare.readlines():
  etichete_pentru_validare.append(etichete_input[7])

#verificarea codului pentru preluarea etichetelor pentru validare

#print(len(etichete_pentru_validare)) #-> 5000

# with open('samples.txt', 'w+') as f:
#   for sample in etichete_pentru_validare:
#     f.write(sample)
#     f.write('\n')


#In submisiile anterioare, aplicam fit strict pe setul pentru antrenat
#Ulterior, m-am gandit ca pot aplica si asupra datelor, respectiv etichetelor de validare
#Initial efectuam doua apeluri, al doilea dupa ce afisam si scorul pentru predictia pe validation
#Am utilizat acest tip de submisie pentru a studia acuratetea, modelul fiind antrenat doar pe datele de test
#Insa pentru submisiile finale, concatenez seturile de date si cele de validare si antrenez direct
#Am observat ca acuratetea creste in acest mod

date_totale_pentru_antrenat = date_pentru_antrenat + date_pentru_validare
etichete_totale_pentru_antrenat = etichete_pentru_antrenat + etichete_pentru_validare

#pentru a constitui parametrii valizi pentru un model, sample urile trebuie formatate
#in primul rand, voi formata etichetele

#pentru a ma asigura ca inputul este parsat corect, creez aceasta lista intermediara
etichete_finale_antrenament = []
for eticheta in etichete_totale_pentru_antrenat:
  etichete_finale_antrenament.append(eticheta[0])

#pe care este necesar sa o transform intr-un array pentru a o utiliza drept parametru

import numpy as np

etichete_finale = np.array(etichete_finale_antrenament)

#cat despre sample uri, voi utiliza un vectorizer pentru a le formata
from sklearn.feature_extraction.text import CountVectorizer

#Am decis, pentru submisiile finale, sa folosesc cate un vectorizer diferit
#Am utilizat si n_grams (valorile au fost alese in urma mai multor testari, incluse in documentatie)
transforma_prin_vectorizare = CountVectorizer(ngram_range=(1,2))

#CountVectorizer dispune de functiile fit_transform si fit
#Voi utiliza fit_transform pe datele finale pentru antrenat
#Aceasta functie presupune invatarea vocabularului si returneaza o matrice

antreneaza_pe = transforma_prin_vectorizare.fit_transform(date_totale_pentru_antrenat)

#print(antreneaza_pe.shape)
#Mi-am afisat shape-ul datelor pe care voi antrena pentru a ma asigura ca este valid
#Prima valoare este 15000: 10000 (initial train data) + 5000 (initial validation data)

testeaza_pe = transforma_prin_vectorizare.transform(date_pentru_testare)

#print(testeaza_pe.shape)
#Si datele pentru testare sunt valide, prima valoare fiind 5000

#Voi importa primul model folosit
from sklearn.naive_bayes import MultinomialNB

primul_optiune = MultinomialNB(alpha=0.02)
#am instantiat modelul cu parametrul alpha
#am inclus in documentatie testele anterioare
#observatii: cu cat alpha este mai mic, cu atat acuratetea este mai mare
#mentiune: pentru 0.02 am obtinut valori mai bune decat 0.01, 0.0001
#am ales la nivel de studiu pe teste
#valoarea default este 1.0, am testat si pentru valori mai mari
#alpha reprezinta parametrul pentru operatia de smooth (netezire) a datelor

#apelez fit pentru a aplica clasificatorul
primul_optiune.fit(antreneaza_pe, etichete_finale)

#dupa ce am efectuat antrenarea, apelez predict pe datele de test
#pentru a efectua predictia
predictie_pe_prima_optiune = primul_optiune.predict(testeaza_pe)

#voi scrie datele optiune din predictie intr-un fisier csv, conform formatului cerut

pred_indx = 0

with open('submission.csv', 'w+') as sub_fis:
    sub_fis.write('id')
    sub_fis.write(',')
    sub_fis.write('label')
    sub_fis.write('\n')
    for pred_id in ids_pentru_testare:
        sub_fis.write(pred_id.split()[0])
        sub_fis.write(',')
        sub_fis.write(str(predictie_pe_prima_optiune[pred_indx]))
        sub_fis.write('\n')
        pred_indx += 1


# In[ ]:





# In[ ]:




