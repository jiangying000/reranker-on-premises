from BCEmbedding import RerankerModel

# your query and corresponding passages
query = 'Is Elisa a good person?'
passages = ["Yes, she is a good person", "Elisa是个好人", "Yes, Elisa is a good person","Elisa is a good person"]

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
