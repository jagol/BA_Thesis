import fasttext

# Skipgram model
model = fasttext.skipgram('data.txt', 'model', thread=1)
print(model.words) # list of words in dictionary