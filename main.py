import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle

df = pd.read_csv('C:\\Users\\gaura\\PycharmProjects\\FAKENEWS\\Symptom2Disease.csv')

x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.1, random_state=5, shuffle=True)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)

vec_train = tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test = tfidf_vectorizer.transform(x_test.values.astype('U'))

pac = PassiveAggressiveClassifier(max_iter=150)
pac.fit(vec_train, y_train)
test_pred = pac.predict(vec_test)

print(f"Test Set Accuracy : {accuracy_score(y_test, test_pred) * 100} %\n\n")
user_input=input("What are the side affects you are experiencing?")
vec_input_test = tfidf_vectorizer.transform([user_input])
result = pac.predict(vec_input_test)
print(result)

pickle.dump(pac,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
pickle.dump(tfidf_vectorizer,open('vectorizer.pkl','wb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))
