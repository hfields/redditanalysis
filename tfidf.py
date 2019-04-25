from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

S1 = "The car is driven on the road."
S2 = "The truck is driven on the highway."

def doit():
    #vectorizer = TfidfVectorizer()
    #response = vectorizer.fit_transform([S1, S2])
    vectorizer = CountVectorizer()
    response = vectorizer.fit_transform([S1, S2])
    print(vectorizer.get_feature_names())
    print(vectorizer.get_stop_words())

    return response