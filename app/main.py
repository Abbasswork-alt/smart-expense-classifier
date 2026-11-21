import sys
from pathlib import Path
import streamlit as st

PROJECT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))
from src.predict import predict

st.set_page_config(page_title='Violet | Expense Classifier', page_icon='🟣', layout='wide')
EXAMPLES = [
    'Carrefour groceries', 'Starbucks coffee', 'McDonalds lunch',
    'Whole Foods groceries', 'Uber ride to airport', 'Shell fuel payment',
    'Delta Air Lines ticket', 'Water utility payment', 'Internet bill Comcast',
    'Netflix monthly payment', 'Spotify subscription', 'Disney+ subscription',
    'Amazon headphones purchase', 'Walmart household shopping',
    'Target clothing purchase', 'Dental clinic appointment', 'Doctor medical visit',
    'Monthly rent payment', 'Microsoft 365 subscription', 'Hilton hotel booking',
]

st.markdown('''
<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(130deg,#f4efff,#ffffff,#edf5ff); color:#25213d;}
.block-container {max-width:1120px; padding-top:2.5rem;}
[data-testid="stSidebar"] {background:#eee8fa;}
[data-testid="stHeader"] {background:transparent;}
h1,h2,h3,p,label {color:#25213d;}
.hero {border-left:6px solid #f3c944; padding:.4rem 1.5rem; margin-bottom:2rem;}
.hero .eyebrow {color:#6941c6; font-weight:800; letter-spacing:.14em; font-size:.8rem;}
.hero h1 {font-size:clamp(2rem,4vw,3.2rem); letter-spacing:-.04em;}
.hero p {color:#625d78; max-width:650px; font-size:1.08rem;}
.pill {display:inline-block; background:#e5efff; color:#2459ae; border-radius:20px; padding:.35rem .8rem; margin:.2rem; font-size:.8rem;}
div.stButton > button[kind="primary"] {background:#6d42c8; color:white; border:0; border-radius:12px; padding:.7rem 1.2rem;}
div.stButton > button[kind="primary"]:hover {background:#5530a5; color:white;}
</style>
<div class="hero"><div class="eyebrow">VIOLET · AI &amp; ML INTERNSHIP</div>
<h1>Small descriptions.<br>Clearer spending.</h1>
<p>Turn an expense description into a category. Explore a prediction and see how confident the model is.</p>
<span class="pill">7 categories</span><span class="pill">English descriptions</span><span class="pill">Local learning demo</span></div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.header('Expense categories')
    st.write('Food · Transport · Bills · Shopping · Health · Entertainment · Other')
    st.divider()
    st.subheader('How to try it')
    st.write('1. Choose an example or write your own.\n\n2. Click Classify expense.\n\n3. Review the category and confidence.')
    st.caption('Each description receives one category. Use separate descriptions for separate purchases.')

def clear_result():
    st.session_state.pop('last_result', None)

def choose_example():
    if st.session_state.example_choice:
        st.session_state.description = st.session_state.example_choice
        clear_result()

left, right = st.columns([1.4,1], gap='large')
with left:
    st.subheader('What did you spend on?')
    st.selectbox('Try one of 20 examples', [''] + EXAMPLES,
        format_func=lambda value: value or 'Choose an example…',
        key='example_choice', on_change=choose_example)
    description = st.text_input('Expense description', placeholder='e.g. Netflix monthly payment',
        key='description', on_change=clear_result)
    if st.button('Classify expense', type='primary', use_container_width=True):
        if not description.strip():
            clear_result()
            st.warning('Enter a description first.')
        else:
            try:
                with st.spinner('Reading your expense…'):
                    st.session_state.last_result = predict(description)
            except FileNotFoundError:
                st.error('The saved model is missing. Restore it in the models folder to continue.')
with right:
    st.subheader('Your prediction')
    with st.container(border=True):
        result = st.session_state.get('last_result')
        if result:
            st.caption('PREDICTED CATEGORY')
            st.subheader(result['category'])
            st.metric('Estimated confidence', f"{result['confidence']:.1%}")
            st.progress(float(result['confidence']))
            st.caption('Confidence is an estimate, not a guarantee.')
        else:
            st.info('Your result will appear here. Choose an example or enter your own expense.')
st.divider()
with st.expander('About this demo & its limitations'):
    st.write('Built with TF-IDF and a calibrated Linear SVM, trained on synthetic English transactions.')
    st.write('Unfamiliar merchants, real descriptions, and combined purchases may be misclassified, even at high confidence. Other includes expenses outside the six specific categories.')
    st.caption('Known limitation: “Uber ride to airport” can be predicted as Other. Suggestions demonstrate model behavior; they are not guaranteed correct answers.')
