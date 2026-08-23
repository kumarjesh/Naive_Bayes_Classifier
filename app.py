import streamlit as st
import joblib

# Configure the visual layout of the app
st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="centered")

# Sidebar - Theme Toggle
with st.sidebar:
    st.header("⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=True)

# Dynamic Theme CSS Injection
if dark_mode:
    st.markdown("""
    <style>
    /* Dark Theme Styles */
    .stApp, [data-testid="stHeader"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #F8FAFC !important;
    }
    div[data-baseweb="textarea"] textarea, textarea {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
    }
    div[data-testid="stButton"] button {
        background-color: #334155 !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #475569 !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #60A5FA !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    /* Light Theme Styles */
    .stApp, [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #0F172A !important;
    }
    div[data-baseweb="textarea"] textarea, textarea {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
    }
    div[data-testid="stButton"] button {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #E2E8F0 !important;
        color: #0F172A !important;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #2563EB !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache the loading process so the backend doesn't reload the model on every button click
@st.cache_resource
def load_models():
    vec = joblib.load('vectorizer.pkl')
    mod = joblib.load('spam_model.pkl')
    return vec, mod

# Initialize models
try:
    vectorizer, model = load_models()
except FileNotFoundError:
    st.error("Backend Error: 'vectorizer.pkl' or 'spam_model.pkl' not found. Please run model training first.")
    st.stop()

# Build the Interface
st.title("🛡️ Email & SMS Spam Detector")
st.markdown("Enter a text message or email below to evaluate whether it is **Spam** or **Ham (Safe)** in real-time.")

# Initialize session state for message input
if "message_text" not in st.session_state:
    st.session_state.message_text = ""

def load_example(text):
    st.session_state.message_text = text

# Preset example buttons for quick testing
st.markdown("##### 💡 Try an Example:")
col_ex1, col_ex2 = st.columns(2)

col_ex1.button(
    "📩 Load Spam Example", 
    on_click=load_example, 
    args=("WINNER! You have won a $1,000 gift card! Claim now by clicking here.",),
    use_container_width=True
)
col_ex2.button(
    "✉️ Load Safe Example", 
    on_click=load_example, 
    args=("Hi Vince, hope you are doing well. Can we schedule a meeting for tomorrow at 3 PM?",),
    use_container_width=True
)

# User Input Area
user_input = st.text_area(
    "Message Content:", 
    key="message_text", 
    height=140, 
    placeholder="Type or paste your message here..."
)

# Execution Logic
if st.button("Analyze Message", type="primary", use_container_width=True):
    if user_input.strip():
        # Transform the raw text using the imported vocabulary
        transformed_input = vectorizer.transform([user_input])
        
        # Execute the Naive Bayes prediction
        prediction = model.predict(transformed_input)[0]
        probabilities = model.predict_proba(transformed_input)[0]
        
        spam_prob = probabilities[1] * 100
        ham_prob = probabilities[0] * 100
        
        st.markdown("---")
        # Display results based on the binary output
        if prediction == 1:
            st.error("🚨 **Alert: This message is classified as SPAM.**")
            st.metric(label="Spam Confidence", value=f"{spam_prob:.2f}%")
        else:
            st.success("✅ **Clear: This message is classified as HAM (Safe).**")
            st.metric(label="Safe Confidence", value=f"{ham_prob:.2f}%")
    else:
        st.warning("Please provide a text message to analyze.")