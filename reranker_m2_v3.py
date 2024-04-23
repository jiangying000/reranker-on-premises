import time

from sentence_transformers import CrossEncoder
import torch

MODEL_NAME = "hotchpotch/japanese-bge-reranker-v2-m3-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder(MODEL_NAME, max_length=512, device=device)
if device == "cuda":
    model.model.half()
query = "感動的な映画について"
passages = [
    "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
    "重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
    "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
    "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
]
def rerank(query: str, documents: list[str]):
    t = time.time()
    scores = model.predict([[query, passage] for passage in documents])
    print(scores)

if __name__ == "__main__":
    rerank(query, passages)