from sentence_transformers import SentenceTransformer
import time
start = time.time()
model = SentenceTransformer('/Users/zesan/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2/snapshots/4328cf26390c98c5e3c738b4460a05b95f4911f5', local_files_only=True)
print("Loaded in", time.time() - start)
