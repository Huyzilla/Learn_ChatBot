from transformers import pipeline

# Tìm kiếm câu trả lời cho câu hỏi từ tài liệu
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def get_answer(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

if __name__ == '__main__':
    test_question = "khóa học nào dạy về Django?"
    test_context = "Khóa học Python nâng cao đi sâu vào lập trình hướng đối tượng (OOP), decorators, và cách xây dựng ứng dụng web với framework Django."
    
    answer = get_answer(test_question, test_context)
    print(f"Câu hỏi: {test_question}")
    print(f"Context: {test_context}")
    print(f"Câu trả lời được trích xuất: {answer}")