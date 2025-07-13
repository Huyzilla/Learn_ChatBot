from retriever import find_best_documents
from reader import get_answer

def answer_my_question(query):
    # Bước 1: Tìm kiếm các tài liệu liên quan
    documents = find_best_documents(query)
    
    # Bước 2: Trả lời câu hỏi dựa trên các tài liệu đã tìm được
    answer = get_answer(query, documents)

    return answer

if __name__ == '__main__':
    user_question = "Khóa học Python có dạy về Django không?"
    final_answer = answer_my_question(user_question)
    print(f"Câu trả lời cuối cùng: {final_answer}")
    
    print("\n" + "="*20 + "\n")

    user_question_2 = "khi nào thì tôi được cấp chứng chỉ?"
    final_answer_2 = answer_my_question(user_question_2)
    print(f"Câu trả lời cuối cùng: {final_answer_2}")
