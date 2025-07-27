import nltk
nltk.download('punkt')
nltk.data.path.append('C:\\\\Users\\\\skandr store\\\\AppData\\\\Roaming\\\ltk_data\\\\tokenizers\\\\punkt')
nltk.download('stopwords')
import textdistance
import re
import time
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from difflib import SequenceMatcher
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize
from langdetect import detect
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
qa_pairs_ar = {
    "كيف أحجز مناسبة؟":
    "\u202B📅 اضغط على 'Start Planning' واملأ التفاصيل للحجز \u202C.",
    "أين أجد كل الايفينتات؟": 
    "\u202B📋 من خلال الضغط على Events هتلاقي كل المناسبات اللي حضرتك سجلتها\u202C.",
    "كيف أضيف خدمات زي ميكب أو تصوير او بدلة؟":
    "💄 بعد اختيار نوع المناسبة، هتقدر تختار الخدمات المطلوبة.",
    "كيف أشغل الابلكيشن؟": 
    "\u202B🔧 افتح التطبيق واضغط Start Planning وابدأ على طول\u202C.",
    "كيف أعدل أو ألغي الحجز ؟": 
    "\u202B 📌 من My Events تقدر تعدل أو تلغي بسهولة \u202C.",
    "مش عارف أبدأ":
    "\u202B✨ مفيش مشكلة! اضغط Start Planning وامشي خطوة بخطوة\u202C."}
qa_pairs_en = {
    "How do I book an event?": "📅 Press 'Start Planning' in website 'EventEase' and fill in the event details.",
    "Where can I find my events?": "📋 Click on 'Events' in website 'EventEase' to see all your booked events.",
    "How do I add services like makeup or photography?": "💄 After selecting the event type, you can choose services like makeup or photography.",
    "How to use the app?": "🔧 Open the app and click 'Start Planning'  in website 'EventEase' to begin.",
    "How to cancel or edit a booking?": "📌 Go to 'My Events' in website 'EventEase' to cancel or edit your booking.",
    "I don't know how to start": "✨ No problem! Just click 'Start Planning' in website 'EventEase' and follow the steps."}
dialect_map_ar = {
    "ازاي": "كيف",
    "ليه": "لماذا",
    "ايه": "ما",
    "عايز": "أريد",
    "عاوزه": "أريد",
    "احجز": "حجز",
    "حجزت": "حجز",
    "ايفينتس": "الايفينتس",
    "ايفينت": "الايفينت",
    "ألغى": "الغي",
    "فين": "أين",
    "إلغاء": "الغاء"}
dialect_map_en={
    "wya": "where are you",
    "btw": "by the way",
    "u": "you",
    "r": "are",
    "idk": "I don't know"}
keywords_groups_ar = {
    "استخدام": ["استخدم", "اشغل","ابدأ","التطبيق","مشغل","مش عارف اشغل","مش عارفة","مشكلة","مشكله", "الابلكيشن"],
    "حجز_مناسبة": ["احجز","حجز","مناسبة", "فرح", "حفلة", "تخرج", "عيدميلاد","كتب كتاب","خطوبة","زفاف","عيد ميلاد","قاعة","مؤتمرات", "حجز لمناسبة"],
    "خدمات": ["فستان", "بيوتي سنتر", "حلاق","بدلة","بدله", "ميكب", "فوتوجرافر","كوافيرة","ضيف","ضيوف","فوتوغرافيا","كوافير","دعوة","دعوات","خدمة","خدمات","ميكب ارتيست","البدلة","دي جي", "تصوير","حجز ميكب", "احجز ميكب"],
    "الايفينتس": ["الايفينت","الايفينتات","قائمة","المناسبات","اللي فاتت","سجلت","مناسباتي","معاد","مواعيد", "events", "evt"],
    "حجز_إداري": ["الغاء حجز","الغي حجز","اكد","اعدل","إلغاء", "ألغي", "الغاء", "ألغيه", "أعدل", "أعدله", "أعدلها", "ألغوها", "إلغاؤه", "إلغاءها", "إلغيه", "احذف","تعديل", "عدلت", "أعدل", "غير", "غيّر","احدث","اغير","تغيير","شيل الحجز","اشطب","تحديث","تعديل حجز","الغي", "تم الحجز", "تأكيد الحجز"]
}
keywords_groups_en = {
    "admin_booking": ["edit","confirm","confirm booking","modify","delete","change","update","cancel","done"],
    "book_event": ["book event","book","venue","birthday", "wedding","party","hall","Conference","Marriage", "graduation", "engagement","create event","reserve","booking","schedule","register"],
    "events": ["events","my events","registered","events list","list","past","date","registered","Occasions","Dates"],
    "services": ["makeup", "photography", "dress","add","Makeup Artist","services","Hairdresser","Barber","Invitation","Invitations","Guest","Guests","dj","decoration","service","catering", "suit", "beauty center", "photographer"],
    "usage": ["how to use","app usage","help","use","run","application","I don't know how to run","Problem","I don't know","start","app","guide","using the app", "how does it work", "open the app"]
}
responses_ar = {
    "استخدام": "🤖 ماذا تريد أن تفعله لكي أساعدك؟",
    "حجز_مناسبة": 
    "\u202B📅 دي مهمة سهلة! اضغط 'Start Planning' واملا بيانات المناسبة\u202C.",
    "خدمات": 
    "\u202B💄 احجز خدماتك بعد ملء تفاصيل المناسبة في 'Start Planning' في ويب سايت 'EventEase' \u202C.",
    "الايفينتس":
    "\u202B📋 تقدر تشوف كل الايفينتات من خلال الضغط على 'Events'  في ويب سايت 'EventEase'\u202C.",
    "حجز_إداري":
    "\u202B📌 للتحكم في الحجوزات (تعديل / إلغاء)، توجه لقسم 'My Events'  واختر الإجراء المناسب \u202C."}
responses_en = {
    "usage": "🔧 Open the app and click 'Start Planning' to begin in website 'EventEase'.",
    "book_event": "📅 Easy! Just click 'Start Planning' in website 'EventEase' and fill in the event details.",
    "services": "💄 After selecting your event, you can add services like makeup or photography.",
    "events": "📋 You can view all your events by clicking on 'Events' in website 'EventEase'.",
    "admin_booking": "📌 To edit or cancel a booking, go to 'My Events' in website 'EventEase' and choose the option you need."}
abbreviations_map = {
    "evt": "الايفينت",
    "evts": "الايفينتات",
    "resv": "حجز",
    "bday": "عيدميلاد",
    "grad": "تخرج",
    "app": "الابلكيشن",
    "svc": "خدمات",
    "mkp": "ميكب",
    "ph": "فوتوجرافر"}
vectorizer_ar = TfidfVectorizer()
question_vectors_ar = vectorizer_ar.fit_transform(list(qa_pairs_ar.keys()))
vectorizer_en = TfidfVectorizer()
question_vectors_en = vectorizer_en.fit_transform(list(qa_pairs_en.keys()))
stopwords_ar = set(stopwords.words("arabic"))
stop_words_en = set(stopwords.words('english'))
all_known_words = set()
stemmer = ISRIStemmer()
for group in list(keywords_groups_ar.values()) + list(keywords_groups_en.values()):
    all_known_words.update(group)
all_known_words.update(dialect_map_ar.keys())
all_known_words.update(dialect_map_en.keys())
all_known_words.update(abbreviations_map.keys())
def detect_language(text):
    try:
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        arabic_ratio = len(arabic_chars) / max(len(text), 1)
        if arabic_ratio > 0.3:
            return 'ar'
        language = detect(text)
        return 'ar' if language == 'ar' else 'en'
    except:
        return 'en'
def normalize_dialect(text):
    language = detect_language(text)
    if language == "ar":
        for word, replacement in dialect_map_ar.items():
            text = text.replace(word, replacement)
    elif language == "en":
        for word, replacement in dialect_map_en.items():
            text = text.replace(word, replacement)
    return text
def normalize_abbreviations(text):
    for abbr, full in abbreviations_map.items():
        text = text.replace(abbr, full)
    return text
def correct_word(word):
    best_match = word
    highest_score = 0
    for known_word in all_known_words:
        score = textdistance.jaro_winkler(word, known_word)
        if score > highest_score and score > 0.85:
            highest_score = score
            best_match = known_word
    return best_match
def correct_input_text(text):
    words = text.split()
    corrected_words = [correct_word(word) for word in words]
    return " ".join(corrected_words)   
def clean_input_keep_stopwords(user_input):
    words = user_input.split()
    keywords_in_input = []
    for word in words:
        if word not in stopwords_ar:
            keywords_in_input.append(word)
        else:
            keywords_in_input.append(word)
    return " ".join(keywords_in_input)
def apply_stemming(text):
    words = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in words]
    return " ".join(stemmed)  
def preprocess_text(text):
    language=detect_language(text)
    text = normalize_dialect(text)
    text = normalize_abbreviations(text)
    text = correct_input_text(text)
    text = clean_input_keep_stopwords(text)
    if language == "ar":
        text = apply_stemming(text)
    return text
qa_pairs_ar = {preprocess_text(k): v for k, v in qa_pairs_ar.items()}
questions_ar= list(qa_pairs_ar.keys())
answers_ar = list(qa_pairs_ar.values())
qa_pairs_en = {preprocess_text(k): v for k, v in qa_pairs_en.items()}
questions_en = list(qa_pairs_en.keys())
answers_en = list(qa_pairs_en.values())
keywords_groups_ar = {
    cat: [preprocess_text(p) for p in phrases]
    for cat, phrases in keywords_groups_ar.items()}
keywords_groups_en = {
    cat: [preprocess_text(p) for p in phrases]
    for cat, phrases in keywords_groups_en.items()}
def smart_reply(user_input, language):
    user_input_processed = preprocess_text(user_input)  
    user_embedding = get_embedding(user_input)
    if language == 'ar':
        user_vector_ar = vectorizer_ar.transform([user_input_processed])
        similarity_ar = cosine_similarity(user_vector_ar, question_vectors_ar)
        max_sim_index_ar = similarity_ar.argmax()
        max_sim_score_ar = similarity_ar[0, max_sim_index_ar]
        if max_sim_score_ar > 0.3:
            return answers_ar[max_sim_index_ar]
    else:
        lowered = user_input.lower()
        if any(word in lowered for word in keywords_groups_en['usage']):
            return responses_en["usage"]
        if any(word in lowered for word in keywords_groups_en['book_event']):
            return responses_en["book_event"]
        if any(word in lowered for word in keywords_groups_en['admin_booking']):
            return responses_en["admin_booking"]
        if any(word in lowered for word in keywords_groups_en['services']):
            return responses_en["services"]
        if any(word in lowered for word in keywords_groups_en['events']):
            return responses_en["events"]
        qa_embeddings = [get_embedding(q) for q in questions_en]
        similarities = [get_similarity(user_embedding, emb) for emb in qa_embeddings]
        max_sim_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        max_sim_score = similarities[max_sim_idx]
        if max_sim_score > 0.65:
            return answers_en[max_sim_idx]
    return None    
def split_long_phrase(text):
    try:
        return sent_tokenize(text)
    except:
        return [text]  
def get_embedding(text):
    return embedding_model.encode(text)
def get_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]
def is_sarcastic(text):
    arabic_patterns = [
        "هو التطبيق بيشتغل", "يعني أضغط الزر", "بالنية", "زرار سحري",
        "يشتغل لوحده", "كل حاجة تتحل", "بلمسة واحدة", "يا سلام",
        "أكيد التطبيق", "أكيد دي", "بإذن الله التطبيق","أكيد الابلكيشن","بإذن الله الابلكيشن", "هو فين الزرار السحري",
        "عايز معجزات", "آه طبعا", "عدم المساعدة", "واو",
        "يا سلام على السهولة"]
    arabic_keywords = [
        "يا سلام", "زرار سحري", "واو","لوحده","الجنة","النار","القمر","ياسلام","زر","زرار","ساحر","النية", "آه طبعا", "طبعا"]
    english_patterns = [
        "oh sure", "of course it works", "magic button", "just like that", "by itself",
        "press a button and done", "god will do it", "it fixes everything", "wow so easy",
        "clearly the app", "wow amazing", "i just click and magic happens"]
    english_keywords = [
        "oh sure", "wow", "magic", "obviously","heaven","hell","magicain","button","moon", "right", "clearly"]
    text = text.lower()
    for pattern in arabic_patterns + english_patterns:
        if pattern in text:
            return True
    for word in arabic_keywords + english_keywords:
        if word in text:
            return True
    return False
def detect_sarcasm(user_input):
    if is_sarcastic(user_input):
        language = detect_language(user_input)
        if language == "en":
            return "😏 Sounds a bit sarcastic! Want real help with something?"
        else:
            return "😏 شكلك بتتكلم بسخرية! هل في حاجة أقدر أساعدك فيها فعلاً؟"
    return None 
def detect_questions(user_input, language='ar'):
    question_separators = ['؟','،', '?', 'و','ثم','كمان','برضو','كذلك','و كمان','و بعدين', 'and', 'also', 'as well', 'then']
    segments = re.split('|'.join(map(re.escape, question_separators)), user_input)
    segments = [s.strip() for s in segments if len(s.strip()) > 3]
    return segments 
def process_single_question(user_input):
    language = detect_language(user_input)
    user_input_norm = normalize_dialect(user_input)
    if language=="ar":
        user_input_norm = normalize_abbreviations(user_input_norm)
        user_input_norm = correct_input_text(user_input_norm)
        user_input_norm = clean_input_keep_stopwords(user_input_norm)
        user_input_norm = apply_stemming(user_input_norm)
    sarcasm_response = detect_sarcasm(user_input)
    if sarcasm_response:
        return sarcasm_response 
    thanks_keywords = ["شكرا", "شكرًا", "متشكر", "thx", "thanks", "thank you"]
    if any(kw in user_input.lower() for kw in thanks_keywords):
        return( "🌟 العفو! سعيد بمساعدتك 😊" if language == "ar" else "🌟 You're welcome! Happy to help 😊")     
    okay_keywords = [ "اشطا", "تمام", "okay"]
    if any(kw in user_input.lower() for kw in okay_keywords):
        return( "🌟  سعيد بمساعدتك 😊" if language == "ar" else "🌟 Happy to help 😊")
    sentences=split_long_phrase(user_input_norm)
    user_embedding = get_embedding(user_input_norm)
    combined_keywords = {**keywords_groups_ar, **keywords_groups_en}
    responses_dict = responses_ar if language == "ar" else responses_en
    cancel_words = ["الغاء", "الغي","إلغاء", "ألغي", "ألغيه", "ألغوها", "إلغاؤه", "إلغاءها", "إلغيه", "ألغى", "احذف", "حذف", "remove", "cancel", "delete"]
    positive_words = ["تأكيد الحجز", "تم الحجز", "confirmed","confirm", "booked"]  
    is_cancel_request = any(word in user_input_norm for word in cancel_words) and not any(word in user_input_norm for word in positive_words)
    if is_cancel_request:
        cancel_targets = (
            keywords_groups_ar["حجز_مناسبة"] + keywords_groups_ar["خدمات"] + keywords_groups_ar["الايفينتس"]
            if language == "ar"
            else keywords_groups_en["book_event"] + keywords_groups_en["services"] + keywords_groups_en["events"]
        )
        if any(t in user_input_norm for t in cancel_targets):
            return responses_ar["حجز_إداري"] if language == "ar" else responses_en["admin_booking"]
    editing_keywords =keywords_groups_en['admin_booking']+ keywords_groups_ar['حجز_إداري']
    editing = any(word in user_input_norm for word in editing_keywords)
    if editing :
        return responses_en["admin_booking"] if language == "en" else responses_ar["حجز_إداري"]    
    def is_generic(response):
        generic_keywords = ["لمساعدتك", "أستطيع مساعدتك", "اسألني", "مساعد", "help", "assist", "anything else"]
        return any(kw in response.lower() for kw in generic_keywords)        
    all_responses=[]    
    seen_responses = set()
    for sentence in sentences:
        processed_sentence = preprocess_text(sentence)
        smart=smart_reply(processed_sentence,language)
        if smart and len(smart.split()) > 2 and not is_generic(smart):
            if smart not in seen_responses:
                all_responses.append(smart)
                seen_responses.add(smart)    
    if all_responses:
        return "\n---\n".join(all_responses)        
    tfidf_responses = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if language == 'ar':
            sentence_vector_ar = vectorizer_ar.transform([sentence])
            similarity_ar = cosine_similarity(sentence_vector_ar, question_vectors_ar)
            max_sim_index_ar = similarity_ar.argmax()
            max_sim_score_ar = similarity_ar[0, max_sim_index_ar]
            if max_sim_score_ar > 0.4:
                tfidf_responses.append(list(qa_pairs_ar.values())[max_sim_index_ar])
        else:
            sentence_vector_en = vectorizer_en.transform([sentence])
            similarity_en = cosine_similarity(sentence_vector_en, question_vectors_en)
            max_sim_index_en = similarity_en.argmax()
            max_sim_score_en = similarity_en[0, max_sim_index_en] 
            if max_sim_score_en > 0.45 :
                tfidf_responses.append(list(qa_pairs_en.values())[max_sim_index_en])
    tfidf_responses = list(dict.fromkeys(tfidf_responses))  
    tfidf_filtered = []
    tfidf_embeddings = []
    for r in tfidf_responses:
        if is_generic(r.strip()):
            continue
        emb = get_embedding(r)
        if any(cosine_similarity([emb], [prev])[0][0] > 0.87 for prev in tfidf_embeddings):
            continue
        tfidf_filtered.append((r, emb))
        tfidf_embeddings.append(emb)
    if tfidf_filtered:
        tfidf_filtered.sort(key=lambda x: cosine_similarity([user_embedding], [x[1]])[0][0], reverse=True)
        top_responses = [r for r, _ in tfidf_filtered[:3]]
        for r in top_responses:
            if r not in seen_responses:
                all_responses.append(r)
                seen_responses.add(r)
        return "\n---\n".join(all_responses)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        category_scores = {}
        for category, phrases in combined_keywords.items():
            matched_keywords = [phrase for phrase in phrases if phrase.lower() in sentence.lower()]
            if matched_keywords:
                priority_weight = 2 if category.lower() in ["book_event", "events", "حجز_مناسبة", "الايفينتس"] else 1
                score = sum(2 if len(phrase.split()) > 1 else 1 for phrase in matched_keywords)
                total_score = score * priority_weight
                category_scores[category] = category_scores.get(category, 0) + total_score 
        if category_scores:
            sorted_category = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            for category, score in sorted_category:
                if category in ["خدمات", "services"]:
                    stronger_categories = [cat for cat, s in sorted_category if s >= score and cat not in ["خدمات", "services"]]
                    if stronger_categories:
                        continue
                if category in responses_dict:
                    if responses_dict[category] not in seen_responses:
                        all_responses.append(responses_dict[category])
                        seen_responses.add(responses_dict[category])
                    break 
    if all_responses:
        return "\n---\n".join(all_responses)            
    return "🤔 مش فاهم عليك كويس، ممكن توضح أكتر؟" if language == "ar" else "🤔 I didn't quite understand. Could you clarify?"
def get_response(user_input):
    language = detect_language(user_input)
    normalized_input = normalize_dialect(user_input)
    arabic_greetings = {"مرحبا","السلام عليكم", "اهلا", "هلا","مساء الخير","صباح الخير","إزيك","صباح الفل يا روبوت","مساء الفل يا روبوت", "هاي", "هيلو", "هالو"}
    english_greetings = {"hello", "hi", "hey", "hallo", "howdy","hey chatbot","good evening","good morning"}
    if language == "ar" and normalized_input.strip() in arabic_greetings:
        return "🌟 هاي! أنا هنا علشان أساعدك تخطط مناسبتك بسهولة 😊"
    elif language == "en" and normalized_input.strip().lower() in english_greetings:
        return "🌟 Hey there! I'm here to help you plan your event with ease 😊"
    questions = detect_questions(user_input, language)

    if len(questions) == 1:
        return process_single_question(questions[0])
    else:
        final_responses = []
        seen = set()
        for q in questions:
            response = process_single_question(q)
            if response:
                for part in response.split("\n---\n"):
                    if part.strip() and part not in seen:
                        final_responses.append(part.strip())
                        seen.add(part.strip())
        if final_responses:
            return "\n---\n".join(final_responses)
        return "🤔 مش فاهم عليك كويس، ممكن توضح أكتر؟" if language == "ar" else "🤔 I didn't quite understand. Could you clarify?"
def type_effect(text, delay=0):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  
def welcome_message():
    type_effect("🎉\u202B أهلاً وسهلاً بيك في شات بوت Event Ease!\u202C")
    type_effect("🎉 Welcome to the Event Ease Chatbot!")
    type_effect("📱 أنا هنا علشان أساعدك تخطط مناسبتك بسهولة – سواء فرح، خطوبة، عيد ميلاد أو غيرهم.")
    type_effect("📱 I'm here to help you easily plan your special events – weddings, engagements, birthdays, and more.")
    type_effect("💡 اسألني عن الحجز، الخدمات، تعديل أو إلغاء المناسبات، أو طريقة استخدام التطبيق.")
    type_effect("💡 Ask me about booking, services, editing or cancelling events, or how to use the app.")
    type_effect("🌐 تقدر تكتب بالعربي أو الإنجليزي.")
    type_effect("🌐 You can talk to me in Arabic or English.")
    print("___________________________")
if __name__ == "__main__":
    welcome_message()
    while True:
        user_input = input("📨 اكتب سؤالك (أو 'خروج' / 'انهاء' / 'باي' للخروج)\n📨 Type your question (or 'exit' / 'bye' to quit): ")
        arabic_goodbyes = ["خروج", "انهاء", "باي","سلام","مع السلامة"]
        english_goodbyes = ["bye", "exit", "goodbye", "see you", "farewell"]
        normalized_input = user_input.strip().lower()
    
        if normalized_input in arabic_goodbyes:
            type_effect("👋\u202B مع السلامة! شكرًا لاستخدامك شات بوت Event Ease \u202C.") 
            break
        elif normalized_input in english_goodbyes:
            type_effect("👋 See you again! Thanks for using the Event Ease Chatbot. Have a great day!")
            break
    
        else:
            response = get_response(user_input)
            type_effect("🤖: " + response)
        
