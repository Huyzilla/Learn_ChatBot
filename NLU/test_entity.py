import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

text = "cho tôi xem thông tin khóa học Python và Machine Learning"

doc = nlp(text)

print("Tokens:")
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")

print("\n--- Thêm Rule để nhận diện 'tên khóa học' ---")
from spacy.matcher import PhraseMatcher # tìm kiếm cụm từ
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
course_names = ['python', 'java', 'machine learning']
patterns = [nlp.make_doc(text) for text in course_names] # Chuyển các từ khóa thành "mẫu" mà spaCy hiểu được
matcher.add("CoursePattern", patterns)

doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print(f" - {span.text} (Loại: ten_khoa_hoc)")


