from sentence_transformers import CrossEncoder

token = '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'


# init reranker model
model = CrossEncoder('maidalun1020/bce-reranker-base_v1', max_length=512)
sentence_pairs =[["I am a good man", "I am a bad person"], ["I am a good teacher", "I am a good person"]]
# calculate scores of sentence pairs
scores = model.predict(sentence_pairs)