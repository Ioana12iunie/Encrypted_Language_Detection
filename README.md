# Encrypted_Language_Detection

### Task

This project consists in my approach to a private Kaggle competition named "Alien Language Classification" I took part in during my second year at the University of Bucharest as part of my Machine Learning Course project. The dataset consisted in short encrypted texts labeled either 1,2,3 based on the language they belonged to. Our task was to classify texts based on the alien language they belong to. We have been encouraged to experiment with available models from sklearn which aligned with our course subjects.

### Overview of approach

The complete documentation can be found [here](https://github.com/Ioana12iunie/Encrypted_Language_Detection/blob/main/Documentatie_ML.pdf).

* In order to extract features, experiements have been conducted using CountVectorizer and Tf-IdfVectorizer.
  
* One of the two submissions we were allowed to use for the final score involved using the MultinomialNB model. The final configuration involved feature extraction with CountVectorizer with n_grams(1,2) and analyzer='char' with an accuracy of 67%.
  
* For the last final submission, I decided to apply one of the observation I was able to make based on previous experiements: higher n-gram ranges provided better results. The model I used was SGDClassifier which performed better than LinearSVC based on the conducted experiemnts. The final configuration involved feature extraction with TfIdfVectorizer with n_grams(1,13) and analyzer='char' for which I obtained a final public score of 75%.

### Conclusions

Classifying encrypted languages and deriving insights from the pursued experiments was truly an engaging and rewarding experience. I found satisfaction in not only applying the theoretical concepts I had learned but also in actively collaborating with the tangible machine learning models. This hands-on experience enriched my understanding and kindled a genuine enthusiasm for the field.
