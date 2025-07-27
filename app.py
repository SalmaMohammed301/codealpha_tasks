import streamlit as st
from event_bot import get_response
from streamlit.components.v1 import html
import time
st.set_page_config(page_title="EventEase Chatbot", page_icon="🤖")

st.markdown("""
<style>
/* الخلفية الموحدة */
html, body, .stApp {
    background: linear-gradient(to bottom right, #1f1c2c, #928dab) !important;
    background-attachment: fixed;
    background-position: center;
    background-size: cover;
    font-family: 'Cairo', sans-serif;
    color: #fff;
    height: 100%;
}

/* إزالة الخلفية البيضاء من chat_input */
.css-1c7y2kd {
    background-color: transparent !important;\
}

/* Animations */
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Message container */
.chat-container {
    display: flex;
    align-items: flex-start;
    margin-bottom: 15px;
    animation: fadeInUp 0.4s ease forwards;
}
.user {
    flex-direction: row-reverse;
}
.chat-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin: 0 10px;
}

/* Message bubbles */
.chat-bubble {
    padding: 14px 18px;
    border-radius: 25px;
    max-width: 75%;
    line-height: 1.6;
    position: relative;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    font-size: 16px;
}

/* User bubble style */
.user .chat-bubble {
    background: linear-gradient(135deg, #b06ab3, #ff6fd8);
    color: white;
    border-radius: 25px 25px 5px 25px;
}
.user .chat-bubble::after {
    content: "";
    position: absolute;
    right: -10px;
    top: 15px;
    width: 0;
    height: 0;
    border-top: 10px solid transparent;
    border-left: 10px solid #b06ab3;
    border-bottom: 10px solid transparent;
}

/* Bot bubble style */
.bot .chat-bubble {
    background: linear-gradient(135deg, #41295a, #2F0743);
    color: white;
    border-radius: 25px 25px 25px 5px;
}
.bot .chat-bubble::after {
    content: "";
    position: absolute;
    left: -10px;
    top: 15px;
    width: 0;
    height: 0;
    border-top: 10px solid transparent;
    border-right: 10px solid #41295a;
    border-bottom: 10px solid transparent;
}

/* Title */
h1 {
    color: #ffffff;
    text-align: center;
    font-size: 2.4em;
    text-shadow: 1px 1px 4px #000;
    margin-bottom: 25px;
}

/* زرار Reset */
#reset-container {
    position: fixed;
    bottom: 1.8rem;
    right: 1.5rem;
    z-index: 9999;
}

</style>
""", unsafe_allow_html=True)

st.title(" 🤖💬 EventEase - Chatbot")

def play_sound():
    html("""
    <audio autoplay>
      <source src="https://www.myinstants.com/media/sounds/iphone-message-swoosh.mp3" type="audio/mpeg">
    </audio>
    """, height=0)
    
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    avatar_url = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png" if role == "user" else "https://cdn-icons-png.flaticon.com/512/4712/4712105.png"
    role_class = "user" if role == "user" else "bot"

    st.markdown(f"""
        <div class="chat-container {role_class}">
            <img class="chat-avatar" src="{avatar_url}" />
            <div class="chat-bubble">{content}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div id="reset-container">
    <form action="" method="post">
        <button type="submit" name="reset" style="background-color:#8467D7; color: white; font-weight: bold; border: none; padding: 10px 20px; border-radius: 10px;">🔁</button>
    </form>
</div>
""", unsafe_allow_html=True)

if st.session_state.get("reset_clicked"):
    st.session_state.messages = []
    st.session_state.reset_clicked = False
    st.rerun()

if "reset" in st.query_params:
    st.session_state.reset_clicked = True
    st.rerun()

user_input = st.chat_input("Enter your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = get_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun() 
play_sound()
