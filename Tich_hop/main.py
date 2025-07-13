from nlu_predictor import predict_nlu
from full_qa_system import full_qa_system

def get_predefined_answer(intent_data):
    intent = intent_data['intent']
    entities = intent_data['entities']

    # Lấy ra tên khóa học nếu có
    course_name = ""
    if entities:
        course_name = entities[0]['value']

    if intent == 'chao_hoi':
        return "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
    elif intent == 'hoi_gia_khoa_hoc':
        if course_name:
            return f"Giá khóa học {course_name} là 2.000.000 VNĐ. Bạn có muốn đăng ký không?"
        else:
            return "Bạn cần hỏi về giá của khóa học nào?"
    return None

def chatbot_train(user_query):
    # Chạy mô hình NLU để lấy intent và entities
    nlu_result = predict_nlu(user_query)
    print(f"NLU Result: {nlu_result}")

    # Lấy câu trả lời từ hệ thống QA
    predefined_answer = get_predefined_answer(nlu_result)

    if predefined_answer:
        return predefined_answer
    else:
        print("Không tìm thấy câu trả lời phù hợp, sử dụng hệ thống QA đầy đủ.")
        return full_qa_system(user_query)
    
    if __name__ == "__main__":
        print("Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")
        while True:
            user_input = input("Bạn: ")
            if user_input.lower() == 'exit':
                break
            
            # Lấy câu trả lời từ "bộ não"
            bot_response = chatbot_brain(user_input)
            print(f"Bot: {bot_response}")
            print("-" * 20)