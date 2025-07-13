# nlu_predictor.py
import joblib
import spacy
from spacy.matcher import PhraseMatcher # tìm kiếm cụm từ

# Tải mô hình phân loại intent
intent_classifier = joblib.load('intent_classifier.pkl')

# Tải mô hình spaCy và thiết lập Matcher cho thực thể
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr='LOWER') 
course_names = ['python', 'java', 'machine learning']
patterns = [nlp.make_doc(text) for text in course_names]
matcher.add("CoursePattern", patterns)

def predict_nlu(text):
    # 1. Dự đoán Intent
    intent = intent_classifier.predict([text])[0]

    # 2. Trích xuất Entity
    doc = nlp(text)
    matches = matcher(doc)
    entities = []
    for match_id, start, end in matches:
        span = doc[start:end]
        entities.append({
            "entity": "ten_khoa_hoc",   
            "value": span.text
        })
    
    # Trả về kết quả NLU
    return {
        "text": text,
        "intent": intent,
        "entities": entities
    }

# Thử nghiệm
if __name__ == '__main__':
    sentence1 = "học phí khóa học python là bao nhiêu"
    prediction1 = predict_nlu(sentence1)
    print(prediction1)
    
    sentence2 = "tôi muốn học về java"
    prediction2 = predict_nlu(sentence2)
    print(prediction2)