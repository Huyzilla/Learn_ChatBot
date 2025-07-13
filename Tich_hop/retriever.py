from sentence_transformers import SentenceTransformer, util
import torch
from knowledge_base import documents

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# encode
documents_embeddings = model.encode(documents, convert_to_tensor=True)

def find_best_documents(query):
    # encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_embedding, documents_embeddings)[0]

    best_doc_index = cos_scores.argmax()

    return documents[best_doc_index]

if __name__ == '__main__':
    test_query = "khóa học nào dạy về Django?"
    best_doc = find_best_documents(test_query)
    print(f"\nCâu hỏi: {test_query}")
    print(f"Tài liệu liên quan nhất: {best_doc}")


