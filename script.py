
# Import TfidfVectorizer
from sklearn import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)


# Computing dot product

import numpy as np

# Initialize numpy vectors
A = np.array([1,3])
B = np.array([-2, 2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)


# Cosine similarity matrix of a corpus


# corpus
# 
# ['The sun is the largest celestial body in the solar system',
#  'The solar system consists of the sun and eight revolving planets',
#  'Ra was the Egyptian Sun God',
#  'The Pyramids were the pinnacle of Egyptian architecture',
#  'The quick brown fox jumps over the lazy dog']



# Import TfidfVectorizer
from sklearn import TfidfVectorizer

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)

print(cosine_sim)

#  [[1.         0.36413198 0.18314713 0.18435251 0.16336438]
#  [0.36413198 1.         0.15054075 0.21704584 0.11203887]
#  [0.18314713 0.15054075 1.         0.21318602 0.07763512]
#  [0.18435251 0.21704584 0.21318602 1.         0.12960089]
#  [0.16336438 0.11203887 0.07763512 0.12960089 1.        ]]

