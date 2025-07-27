import streamlit as st
from deep_translator import GoogleTranslator
import pyperclip
import pyttsx3
import os

st.set_page_config(page_title="🌍  Language Translator", page_icon="🌐")

# 🧩 ستايل
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Cairo', sans-serif;
    color: #e3f2fd;
    position: relative;
    height: 100%;
    overflow-y: auto !important;
    overflow-x: hidden;
}
.stApp::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 0;
    background-image: 
        url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='500' height='500'><text x='10' y='50' font-size='60' fill='white' fill-opacity='0.03'>🌍</text><text x='100' y='150' font-size='60' fill='white' fill-opacity='0.03'>🔤</text><text x='200' y='250' font-size='60' fill='white' fill-opacity='0.03'>💬</text><text x='300' y='100' font-size='60' fill='white' fill-opacity='0.03'>🈳</text><text x='50' y='300' font-size='60' fill='white' fill-opacity='0.03'>📘</text><text x='400' y='400' font-size='60' fill='white' fill-opacity='0.03'>🗣️</text><text x='150' y='450' font-size='60' fill='white' fill-opacity='0.03'>↔️</text></svg>");
    background-repeat: repeat;
    background-size: 500px 500px;
}
.stApp > * {
    position: relative;
    z-index: 1;
}
textarea, .stTextInput, .stSelectbox {
    background-color: rgba(255, 255, 255, 0.07) !important;
    color: #ffffff !important;
    border: 1.5px solid #4fc3f7 !important;
    border-radius: 10px !important;
    padding: 12px;
}
.stButton > button {
    background: linear-gradient(to right, #42a5f5, #1e88e5);
    color: #ffffff;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    margin-top: 12px;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(to right, #1565c0, #0d47a1);
}
h1, h2, h3 {
    color: #81d4fa;
    text-align: center;
    font-weight: 700;
}
.stAlert-success {
    background-color: #388e3c;
    border-left: 5px solid #66bb6a;
    color: #ffffff;
}
textarea[readonly] {
    background-color: rgba(200, 230, 201, 0.05) !important;
    font-style: italic;
    color: #e0f7fa !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🌐✨ Instant Translator")
st.markdown("### Translate texts between different languages easily")

lang_code_map = {
    "af": "Afrikaans - الأفريكانية",
    "am": "Amharic - الأمهرية",
    "ar": "Arabic - العربية",
    "az": "Azerbaijani - الأذربيجانية",
    "be": "Belarusian - البيلاروسية",
    "bg": "Bulgarian - البلغارية",
    "bn": "Bengali - البنغالية",
    "bs": "Bosnian - البوسنية",
    "ca": "Catalan - الكاتالونية",
    "ceb": "Cebuano - السيبوانو",
    "cs": "Czech - التشيكية",
    "cy": "Welsh - الويلزية",
    "da": "Danish - الدنماركية",
    "de": "German - الألمانية",
    "el": "Greek - اليونانية",
    "en": "English - الإنجليزية",
    "eo": "Esperanto - الإسبرانتو",
    "es": "Spanish - الإسبانية",
    "et": "Estonian - الإستونية",
    "eu": "Basque - الباسكية",
    "fa": "Persian - الفارسية",
    "fi": "Finnish - الفنلندية",
    "fr": "French - الفرنسية",
    "fy": "Frisian - الفريزية",
    "ga": "Irish - الأيرلندية",
    "gd": "Scottish Gaelic - الغيلية الأسكتلندية",
    "gl": "Galician - الجاليكية",
    "gu": "Gujarati - الغوجاراتية",
    "ha": "Hausa - الهوسا",
    "haw": "Hawaiian - الهاواي",
    "he": "Hebrew - العبرية",
    "hi": "Hindi - الهندية",
    "hmn": "Hmong - الهمونغ",
    "hr": "Croatian - الكرواتية",
    "ht": "Haitian Creole - الكريولية الهايتية",
    "hu": "Hungarian - المجرية",
    "hy": "Armenian - الأرمنية",
    "id": "Indonesian - الإندونيسية",
    "ig": "Igbo - الإيجبو",
    "is": "Icelandic - الأيسلندية",
    "it": "Italian - الإيطالية",
    "ja": "Japanese - اليابانية",
    "jv": "Javanese - الجاوية",
    "ka": "Georgian - الجورجية",
    "kk": "Kazakh - الكازاخستانية",
    "km": "Khmer - الخمير",
    "kn": "Kannada - الكانادا",
    "ko": "Korean - الكورية",
    "ku": "Kurdish - الكردية",
    "ky": "Kyrgyz - القيرغيزية",
    "la": "Latin - اللاتينية",
    "lb": "Luxembourgish - اللوكسمبورغية",
    "lo": "Lao - اللاوية",
    "lt": "Lithuanian - اللتوانية",
    "lv": "Latvian - اللاتفية",
    "mg": "Malagasy - المالاجاشية",
    "mi": "Maori - الماورية",
    "mk": "Macedonian - المقدونية",
    "ml": "Malayalam - المالايالامية",
    "mn": "Mongolian - المنغولية",
    "mr": "Marathi - الماراثية",
    "ms": "Malay - الملايو",
    "mt": "Maltese - المالطية",
    "my": "Myanmar - البورمية",
    "ne": "Nepali - النيبالية",
    "nl": "Dutch - الهولندية",
    "no": "Norwegian - النرويجية",
    "ny": "Nyanja - النيانية",
    "pa": "Punjabi - البنجابية",
    "pl": "Polish - البولندية",
    "ps": "Pashto - البشتو",
    "pt": "Portuguese - البرتغالية",
    "ro": "Romanian - الرومانية",
    "ru": "Russian - الروسية",
    "rw": "Kinyarwanda - الكينيارواندية",
    "sd": "Sindhi - السندية",
    "si": "Sinhala - السنهالية",
    "sk": "Slovak - السلوفاكية",
    "sl": "Slovenian - السلوفينية",
    "sm": "Samoan - الساموانية",
    "sn": "Shona - الشونا",
    "so": "Somali - الصومالية",
    "sq": "Albanian - الألبانية",
    "sr": "Serbian - الصربية",
    "st": "Sesotho - السوتو",
    "su": "Sundanese - السوندانية",
    "sv": "Swedish - السويدية",
    "sw": "Swahili - السواحيلية",
    "ta": "Tamil - التاميلية",
    "te": "Telugu - التيلوجو",
    "tg": "Tajik - الطاجيكية",
    "th": "Thai - التايلاندية",
    "tk": "Turkmen - التركمانية",
    "tl": "Tagalog - التغالوغية",
    "tr": "Turkish - التركية",
    "tt": "Tatar - التترية",
    "ug": "Uyghur - الأويغورية",
    "uk": "Ukrainian - الأوكرانية",
    "ur": "Urdu - الأردية",
    "uz": "Uzbek - الأوزبكية",
    "vi": "Vietnamese - الفيتنامية",
    "xh": "Xhosa - الخوسية",
    "yi": "Yiddish - اليديشية",
    "yo": "Yoruba - اليوروبية",
    "zh": "Chinese - الصينية",
    "zu": "Zulu - الزولو"
}

with st.sidebar:
    st.markdown("### ℹ️ Language Codes")
    st.caption("اختصارات اللغات المستخدمة في الترجمة:")
    st.caption("Abbreviations of languages used in translation:")
    lang_filter = st.text_input("🔍Search for a language:")
    filtered_langs = {
        code: name
        for code, name in lang_code_map.items()
        if lang_filter.lower() in code.lower() or lang_filter.lower() in name.lower()
    }
    if filtered_langs:
        for code, name in filtered_langs.items():
            st.markdown(f"- **`{code}`**: {name}")
    else:
        st.info("❌ لا توجد نتائج مطابقة")


langs = GoogleTranslator().get_supported_languages(as_dict=True)
lang_names = list(langs.values())

def get_index_of_lang(lang_keyword):
    for i, name in enumerate(lang_names):
        if lang_keyword.lower() in name.lower():
            return i
    return 0

default_src_index = get_index_of_lang("english")
default_tgt_index = get_index_of_lang("arabic")

# 🧠 session_state
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

text = st.text_area("📝 Enter text here:", height=150)
col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("🔤 Source language:", lang_names, index=default_src_index)
with col2:
    target_lang = st.selectbox("🌍 Target language:", lang_names, index=default_tgt_index)

source_lang_code = [k for k, v in langs.items() if v == source_lang][0]
target_lang_code = [k for k, v in langs.items() if v == target_lang][0]

# ترجمة
if st.button("🔄 Translate"):
    try:
        translated = GoogleTranslator(source=source_lang_code, target=target_lang_code).translate(text)
        st.session_state.translated_text = translated
    except Exception as e:
        st.error(f"❌ Error: {e}")

if st.session_state.translated_text.strip():
    st.success("✅ Translated text:")
    st.text_area("📄 Translate:", st.session_state.translated_text, height=150, key="output_text")

    col3 = st.columns(1)[0]
    with col3:
        if st.button("📋 Copy Translate "):
            pyperclip.copy(st.session_state.translated_text)
            st.toast("✔️ Translation copied!")
