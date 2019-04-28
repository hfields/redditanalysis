from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

S1 = "The car is driven on the road."
S2 = "The truck is driven on the highway."

def doit():
    vectorizer = CountVectorizer(stop_words=['car'])
    counts = vectorizer.fit_transform([S1, S2])
    transformer = TfidfTransformer()
    response = transformer.fit_transform(counts)
    print(vectorizer.get_feature_names())
    print(vectorizer.vocabulary_)
    print(response.mean(axis=0))
    print(response.toarray())