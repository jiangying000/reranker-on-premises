from BCEmbedding import RerankerModel

# your query and corresponding passages
query = 'I am a good person?'
passages = ['I am a good man', 'I am a good teacher', 'I am a good person', 'Yes, I am a good person', 'I am a bad person', '我是个好人']

# construct sentence pairs
sentence_pairs = [
    (query, passage) for passage in passages]


# init reranker model
model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")

# method 0: calculate scores of sentence pairs
scores = model.compute_score(sentence_pairs)
import time
# method 1: rerank passages
time1 = time.time()
rerank_results = model.rerank(query, passages)
print("t: ", time.time()-time1)

print(rerank_results)
