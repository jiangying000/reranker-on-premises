import time

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from BCEmbedding import RerankerModel

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class QueryPassages(BaseModel):
    model: str
    query: str
    documents: List


# init reranker model
model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")


@app.post("/rerank")
async def rerank_passages(input_data: QueryPassages):
    t = time.time()
    docs1 = []
    for doc in input_data.documents:
        if isinstance(doc, dict):
            if 'text' not in doc:
                return {"error": "Missing 'text' key in document"}
            docs1.append(doc['text'])
        elif isinstance(doc, str):
            docs1.append(doc)
        else:
            return {"error": "Invalid document format"}
    rerank_results = model.rerank(input_data.query, docs1)
    dt = time.time() - t
    # return {
    #     "time": dt,
    #     "rerank_results": rerank_results
    # }
    results = []
    for passage, score, idx in zip(rerank_results['rerank_passages'], rerank_results['rerank_scores'],
                                   rerank_results['rerank_ids']):
        result = {
            "document": {
                "text": passage
            },
            "index": idx,
            "relevance_score": score
        }
        if isinstance(input_data.documents[idx], dict):
            result['document'] = input_data.documents[idx]
        results.append(result)

    result = {
        "time": dt,
        "model": "maidalun1020/bce-reranker-base_v1",
        "results": results
    }
    return result


#
# {
#   "model": "jina-reranker-v1-base-en",
#   "usage": {
#     "total_tokens": 38,
#     "prompt_tokens": 38
#   },
#   "results": [
#     {
#       "index": 3,
#       "document": {
#         "text": "Natural organic skincare range for sensitive skin"
#       },
#       "relevance_score": 0.8292155861854553
#     },
#     {
#       "index": 2,
#       "document": {
#         "text": "Organic cotton baby clothes for sensitive skin"
#       },
#       "relevance_score": 0.14426936209201813
#     },
#     {
#       "index": 6,
#       "document": {
#         "text": "Sensitive skin-friendly facial cleansers and toners"
#       },
#       "relevance_score": 0.13857832551002502
#     }
#   ]
# }
#
# "results": [
#     {
#         "document": {
#             "text": "string"
#         },
#         "index": 0,
#         "relevance_score": 0
#     }
# ],

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
