"""ViralLab — TikTok-inspired interactive showcase for the original trained model."""
from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="ViralLab • TikTok Intelligence", page_icon="🎵", layout="wide", initial_sidebar_state="expanded")
PINK = "#ff347b"
CYAN = "#38e8ef"
BG = "#080a11"
WHITE = "#f6f8ff"
MUTED = "#aab5cb"

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{color-scheme:dark}
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif!important}
.stApp{background:radial-gradient(ellipse 65% 42% at 95% 0%,rgba(255,52,123,.13),transparent 75%),radial-gradient(ellipse 55% 40% at 10% 45%,rgba(56,232,239,.055),transparent 80%),#080a11!important;color:#f6f8ff!important}
.block-container{padding-top:2rem!important;padding-bottom:4rem!important;max-width:1330px!important}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;color:#f6f8ff!important;letter-spacing:-.035em!important}
p,li,small,label,[data-testid="stCaptionContainer"],.stMarkdown{color:#d6deef!important}
[data-testid="stSidebar"]{background:#0d111c!important;border-right:1px solid #263044!important}
[data-testid="stSidebar"] *{color:#e9efff!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:12px;padding:10px 12px!important;margin:2px 0!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:#202032!important;outline:1px solid #ff347b88}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"]{color:white!important}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-baseweb="select"]>div{background:#151b29!important;color:#f6f8ff!important;border:1px solid #3a4761!important;border-radius:12px!important}
textarea::placeholder,input::placeholder{color:#aab5cb!important;opacity:1!important}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label{color:#eaf1ff!important;font-weight:700!important}
[data-testid="stSelectbox"] *{color:#f6f8ff!important}
[data-baseweb="popover"] *{color:#f6f8ff!important}
[data-testid="stSlider"] *{color:#eaf1ff!important}
[data-testid="stSlider"] [role="slider"]{background:#ff347b!important;border-color:#ff347b!important}
[data-testid="stMetric"]{background:linear-gradient(145deg,#171d2c,#111624)!important;border:1px solid #303d56!important;border-radius:17px;padding:18px!important}
[data-testid="stMetricLabel"] *{color:#b5c4db!important}
[data-testid="stMetricValue"] *{color:#fff!important;font-family:'Space Grotesk',sans-serif!important}
.stButton>button{border:1px solid #47516a!important;background:#192133!important;color:#f7f9ff!important;border-radius:13px!important;font-weight:800!important;padding:.7rem 1rem!important;transition:all .2s}
.stButton>button:hover{border-color:#38e8ef!important;color:#38e8ef!important;transform:translateY(-1px)}
.stButton>button[kind="primary"]{background:linear-gradient(110deg,#ff347b,#d32e9e)!important;color:white!important;border:0!important;box-shadow:0 9px 28px #ff347b30!important}
[data-testid="stAlert"]{background:#172235!important;border:1px solid #354a65!important;border-radius:14px!important;color:#fff!important}
[data-testid="stAlert"] *{color:#e9f5ff!important}
[data-testid="stExpander"]{background:#131a29!important;border:1px solid #303b51!important;border-radius:14px!important}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden}
hr{border-color:#30394a!important}
.vl-hero{padding:36px 38px;border-radius:25px;background:radial-gradient(circle at 92% 15%,#ff347b38,transparent 35%),linear-gradient(125deg,#171b2b,#111a29 60%,#112630);border:1px solid #3c435d;position:relative;overflow:hidden;margin-bottom:24px}
.vl-hero:after{content:'✦';position:absolute;right:40px;top:12px;font-size:95px;color:#38e8ef23}
.vl-eyebrow{color:#38e8ef!important;font-size:12px;font-weight:800;letter-spacing:.2em;text-transform:uppercase;margin:0 0 12px}
.vl-hero h1{font-size:clamp(32px,4vw,57px);line-height:1.06;margin:0 0 12px}
.vl-hero p{font-size:16px;color:#c6d4e7!important;max-width:650px;line-height:1.65;margin:0}
.vl-pink{color:#ff4d94}
.vl-cyan{color:#38e8ef}
.vl-card{border:1px solid #303c54;background:linear-gradient(145deg,#161d2c,#101522);border-radius:19px;padding:23px;margin-bottom:15px;min-height:130px}
.vl-card h3{font-size:19px;margin:9px 0 8px}.vl-card p{color:#bdc9de!important;font-size:14px;line-height:1.5;margin:0}
.vl-icon{font-size:27px}
.vl-pill{display:inline-block;padding:7px 12px;border-radius:100px;border:1px solid #46516b;background:#192234;color:#e5eeff;font-size:12px;font-weight:800;margin:0 7px 8px 0}
.vl-result{border:1px solid #ff347b7a;background:linear-gradient(120deg,#36172b,#171b30);border-radius:20px;padding:22px 26px;margin:15px 0}
.vl-result h2{font-size:30px;margin:8px 0;color:#fff!important}
.vl-result p{color:#e2d8e7!important;margin:0}
.vl-note{font-size:13px;color:#b5c4d8!important;line-height:1.55}
.vl-step{padding:12px 15px;border-radius:12px;background:#151d2c;border:1px solid #2b3a50;color:#dcecff;font-size:13px;font-weight:700;text-align:center;margin:5px 0}
.vl-brand{font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:800;letter-spacing:-.06em;color:#fff}
.vl-divider{height:1px;background:linear-gradient(90deg,#ff347b,#38e8ef,transparent);margin:20px 0}

/* Readable alerts, including Streamlit's pale info/warning panels */
[data-testid="stAlert"], [data-testid="stAlert"] > div {background:#162235!important;color:#f4f7ff!important}
[data-testid="stAlert"] p,[data-testid="stAlert"] span,[data-testid="stAlert"] div{color:#f4f7ff!important}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"]{gap:6px}
[data-testid="stSidebar"] [data-testid="stRadio"] label{font-size:15px!important}
.vl-cluster-chip{display:inline-block;background:#fff!important;color:#10131b!important;border-radius:9px;padding:7px 13px;margin:8px 0 12px;font-size:13px;font-weight:800}
.vl-brand-logo{display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;margin-right:8px;vertical-align:middle;background:#05060b;border-radius:11px;border:1px solid #303849}
.vl-brand-logo img{width:25px;height:25px;filter:drop-shadow(-2px -1px 0 #25f4ee) drop-shadow(2px 2px 0 #fe2c55)}
.vl-explain{padding:18px 21px;border:1px solid #34455e;background:#141e2e;border-radius:15px;color:#e7f0ff!important;margin:12px 0;line-height:1.65}
.vl-explain strong{color:#fff!important}
.vl-step{min-height:72px;display:flex;align-items:center;justify-content:center;font-size:14px}

/* Home: editorial hierarchy instead of repeated boxed cards */
.vl-home-intro{max-width:850px;margin:32px 0 30px;padding-left:22px;border-left:3px solid #fe2c55}
.vl-home-kicker,.vl-home-feature span{color:#25f4ee!important;letter-spacing:.15em;font-size:11px;font-weight:800}
.vl-home-intro h2{font-size:clamp(24px,3vw,35px);margin:10px 0}
.vl-home-intro p{font-size:17px;line-height:1.8;color:#c5d0e0!important;margin:0}
.vl-home-feature{padding:8px 18px 15px 0;min-height:155px}
.vl-home-feature h3{margin:12px 0 8px;font-size:21px}
.vl-home-feature p{color:#bac9dd!important;line-height:1.65;font-size:14px}
.vl-home-rule{height:1px;background:linear-gradient(90deg,#303d51,#303d5155,transparent);margin:24px 0 30px}
.vl-home-journey{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:20px 0 30px}
.vl-home-journey-item{border-top:2px solid #25f4ee;padding:15px 10px 0 0;display:flex;flex-direction:column;gap:9px}
.vl-home-journey-item:nth-child(even){border-top-color:#fe2c55}
.vl-home-journey-item span{font-size:12px;color:#25f4ee;font-weight:800}
.vl-home-journey-item strong{color:#f6f8ff;font-size:15px}
.vl-home-journey-item small{color:#aebbd0;font-size:12px;line-height:1.55}
@media(max-width:850px){.vl-home-journey{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading our TikTok prediction model…")
def load_bundle():
    # Load only the model file exported from your own trusted notebook.
    return joblib.load(Path(__file__).with_name("virallab_models.joblib"))

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"The prediction model could not be loaded: {exc}")
    st.info("Use the same scikit-learn version used when exporting the notebook (1.6.1). Restart Streamlit after changing packages.")
    st.stop()

sentiment_analyzer = SentimentIntensityAnalyzer()

def predict(text: str, seconds: int, verified: str) -> dict:
    """Preserves the original exported NLP → cluster → RF prediction pipeline."""
    if not text.strip():
        raise ValueError("Please enter a transcript or description first.")
    topic_vector = bundle["nmf_model"].transform(bundle["tfidf_vectorizer"].transform([text]))
    topic = bundle["topic_mapping"][int(np.argmax(topic_vector[0]))]
    duration = "Short" if seconds <= 15 else ("Medium" if seconds <= 40 else "Long")
    video = pd.DataFrame({
        "word_count": [len(text.split())],
        "sentiment_score": [sentiment_analyzer.polarity_scores(text)["compound"]],
        "duration_category": [duration],
        "topic_category": [topic],
        "verified_status": [verified],
    })
    k_input = np.column_stack([
        bundle["kproto_scaler"].transform(video[bundle["numeric_features"]]),
        video[bundle["categorical_features"]].astype(str).to_numpy(),
    ])
    cluster = int(bundle["kproto"].predict(k_input, categorical=bundle["categorical_indices"])[0])
    video["KPrototypes_Cluster"] = str(cluster)
    features = bundle["pipe_numeric"] + bundle["pipe_categorical_with_cluster"]
    X = bundle["pipe_preprocess"].transform(video[features])
    idx = list(bundle["pipe_model"].classes_).index("Viral")
    probability = float(bundle["pipe_model"].predict_proba(X)[0][idx])
    threshold = float(bundle["best_t_rf"])
    return {
        "label": "Viral" if probability >= threshold else "Non-Viral",
        "prob": probability,
        "threshold": threshold,
        "topic": str(topic),
        "cluster": cluster,
        "sentiment": float(video["sentiment_score"].iloc[0]),
        "duration": duration,
    }


def hero(eyebrow: str, heading: str, description: str):
    st.markdown(f'<div class="vl-hero"><div class="vl-eyebrow">{html.escape(eyebrow)}</div><h1>{heading}</h1><p>{html.escape(description)}</p></div>', unsafe_allow_html=True)


def card(icon: str, title: str, description: str):
    st.markdown(f'<div class="vl-card"><div class="vl-icon">{icon}</div><h3>{html.escape(title)}</h3><p>{html.escape(description)}</p></div>', unsafe_allow_html=True)


def probability_chart(probability: float, threshold: float, height: int = 295):
    value = max(0, min(100, probability * 100))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "font": {"size": 43, "color": WHITE}},
        title={"text": "MODEL VIRAL SCORE", "font": {"size": 13, "color": MUTED}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": PINK, "thickness": .3},
            "bgcolor": "#151c2b", "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": "#1c2837"}],
            "threshold": {"line": {"color": CYAN, "width": 5}, "thickness": .9, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=45, b=5), paper_bgcolor="rgba(0,0,0,0)", font_color=WHITE)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_result(result: dict, prefix=""):
    label = result["label"]
    st.markdown(f'<div class="vl-result"><div class="vl-eyebrow">ANALYSIS COMPLETE</div><h2>{"Predicted VIRAL" if label == "Viral" else "Predicted NON-VIRAL"}</h2><p>This is our trained model’s classification, not a guarantee of real-world performance.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model viral score", f'{result["prob"]:.1%}')
    c2.metric("Decision threshold", f'{result["threshold"]:.1%}')
    c3.metric("Assigned cluster", f'#{result["cluster"]}')
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        probability_chart(result["prob"], result["threshold"])
    with right:
        st.markdown("### What the model noticed")
        st.markdown(f'<span class="vl-pill">Topic: {html.escape(result["topic"])}</span><span class="vl-pill">Duration: {result["duration"]}</span><span class="vl-pill">Sentiment: {result["sentiment"]:+.2f}</span>', unsafe_allow_html=True)
        st.write("The cyan marker is the model's decision threshold. Scores above that threshold receive the **Viral** label.")
        st.caption("This score is a model estimate, not a calibrated guarantee of TikTok reach.")

st.sidebar.markdown('<div class="vl-brand"><span class="vl-brand-logo"><img src="https://cdn.simpleicons.org/tiktok/ffffff" alt="TikTok logo"></span>Viral<span class="vl-pink">Lab</span></div>', unsafe_allow_html=True)
st.sidebar.caption("TIKTOK INTELLIGENCE • DATA MINING")
st.sidebar.markdown('<div class="vl-divider"></div>', unsafe_allow_html=True)
page = st.sidebar.radio("EXPLORE", ["Home", "Dashboard", "Predict virality", "What-if lab", "Model explorer", "Challenge the model"], label_visibility="collapsed")
st.sidebar.markdown('<div class="vl-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<span class="vl-pill">● Model connected</span>', unsafe_allow_html=True)
st.sidebar.caption("Explore our research model. Results are estimates, not guaranteed TikTok performance.")

if page == "Home":
    hero("The TikTok data mining experience", 'Can data predict <span class="vl-pink">viral</span> content?', "Discover how what you say, how long your video is, and whether an account is verified can affect our model’s prediction of viral potential.")
    st.markdown("""<div class="vl-home-intro"><div class="vl-home-kicker">THE IDEA</div><h2>What makes a TikTok video stand out?</h2><p>Some videos take off while others barely get noticed. Our research explores whether the words used in a video, its length, and whether the account is verified can help our model estimate its viral potential.</p></div>""", unsafe_allow_html=True)
    st.markdown("### Explore ViralLab")
    left, mid, right = st.columns(3, gap="large")
    with left:
        st.markdown('<div class="vl-home-feature"><span>01 / EXPLORE</span><h3>Discover the patterns</h3><p>Explore example video ideas and see how their model scores compare.</p></div>', unsafe_allow_html=True)
    with mid:
        st.markdown('<div class="vl-home-feature"><span>02 / EXPERIMENT</span><h3>Test a video idea</h3><p>Describe a video, adjust its details, and see how our model responds.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="vl-home-feature"><span>03 / PLAY</span><h3>Challenge the model</h3><p>Choose which of two videos will receive the higher model score.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="vl-home-rule"></div>', unsafe_allow_html=True)
    st.markdown("### Our model at a glance")
    a, b, c, d = st.columns(4)
    a.metric("Videos studied", "19,084", help="Count reported in the original notebook, not dynamically reloaded.")
    b.metric("Content themes", int(bundle["nmf_model"].n_components))
    c.metric("Video groups", int(bundle["kproto"].n_clusters))
    d.metric("Prediction cutoff", f'{float(bundle["best_t_rf"]):.0%}')
    st.markdown('<div class="vl-home-rule"></div>', unsafe_allow_html=True)
    st.markdown("### From video idea to prediction")
    steps = [
        ("01", "Video context", "What the video is about and how it is described"),
        ("02", "Content theme", "The main subject of the video"),
        ("03", "Language tone", "The emotional tone of the words"),
        ("04", "Similar patterns", "How it compares with groups in our research"),
        ("05", "Viral potential", "The model's estimated score and classification"),
    ]
    st.markdown('<div class="vl-home-journey">' + ''.join(f'<div class="vl-home-journey-item"><span>{num}</span><strong>{title}</strong><small>{desc}</small></div>' for num,title,desc in steps) + '</div>', unsafe_allow_html=True)
    st.markdown("**What does the score mean?** Our model uses video text, duration, and account verification—not likes, shares, or views—to make an experimental prediction. It cannot guarantee real-world virality.")
    if st.button("Try a live prediction", type="primary", use_container_width=True):
        st.session_state["home_go"] = True
    if st.session_state.get("home_go"):
        st.success("Select **Predict virality** in the sidebar to begin.")

elif page == "Dashboard":
    hero("Interactive insights dashboard", 'Explore <span class="vl-pink">viral potential.</span>', "Compare different TikTok ideas using our trained model. Every score below is generated live from our model — not an actual view or like count.")
    st.markdown("### Explore video ideas")
    st.caption("Choose a topic and account type. Our dashboard tests several example videos with the same length so you can compare the model's predictions fairly.")
    scenarios = {
        "Sports": "An incredible last-minute goal surprised the crowd during an exciting championship match.",
        "Food": "Here is a simple recipe for crispy potatoes with a quick and easy cooking tip.",
        "Technology": "A new artificial intelligence tool helps students learn and organize their studies.",
        "Science": "Scientists discovered a distant planet that could reveal new secrets about the universe.",
        "Lifestyle": "A morning routine with simple habits to feel more organized and productive every day.",
        "Travel": "Explore a beautiful destination and discover the best places to visit on a budget.",
    }
    x,y,z=st.columns([1.1,1,1])
    with x: focus=st.selectbox("Highlight a video idea",list(scenarios),key="dash_focus")
    with y: dash_duration=st.slider("Video length (seconds)",5,60,25,key="dash_duration")
    with z: dash_verified=st.selectbox("Is the account verified?",["not verified","verified"],key="dash_verified")
    @st.cache_data(show_spinner=False)
    def dashboard_predictions(seconds, verified):
        rows=[]
        for name, txt in scenarios.items():
            result=predict(txt,seconds,verified)
            rows.append({"Idea":name,"Model score (%)":round(result["prob"]*100,2),"Prediction":result["label"],"Group":result["cluster"]})
        return pd.DataFrame(rows)
    try:
        with st.spinner("Comparing ideas with our trained model…"):
            dashboard_df=dashboard_predictions(dash_duration,dash_verified)
        selected=dashboard_df.loc[dashboard_df["Idea"]==focus].iloc[0]
        a,b,c=st.columns(3)
        a.metric("Selected idea's model score",f'{selected["Model score (%)"]:.1f}%')
        b.metric("Ideas compared",len(dashboard_df))
        c.metric("Ideas labeled viral",int((dashboard_df["Prediction"]=="Viral").sum()))
        st.markdown("### Compare predicted viral potential")
        ordered=dashboard_df.sort_values("Model score (%)",ascending=True)
        fig=go.Figure(go.Bar(y=ordered["Idea"],x=ordered["Model score (%)"],orientation="h",marker_color=[CYAN if v==focus else PINK for v in ordered["Idea"]],text=[f'{v:.1f}%' for v in ordered["Model score (%)"]],textposition="outside",hovertemplate="%{y}: %{x:.1f}% model score<extra></extra>"))
        fig.add_vline(x=float(bundle["best_t_rf"])*100,line_dash="dash",line_color=CYAN,annotation_text="Viral-label cutoff",annotation_font_color=CYAN)
        fig.update_layout(template="plotly_dark",height=390,xaxis_range=[0,110],xaxis_title="Model score (%)",yaxis_title="",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color=WHITE,margin=dict(l=10,r=30,t=25,b=20),showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.markdown(f'<div class="vl-explain"><strong>{html.escape(focus)} example:</strong> {html.escape(scenarios[focus])}<br><strong>Our model predicts:</strong> {html.escape(str(selected["Prediction"]))} ({selected["Model score (%)"]:.1f}% model score). This is an estimate, not a forecast of actual views.</div>',unsafe_allow_html=True)
        st.markdown("### How much does video length change this prediction?")
        lengths=[10,15,20,30,40,50,60]
        trend=[predict(scenarios[focus],d,dash_verified)["prob"]*100 for d in lengths]
        trend_fig=go.Figure(go.Scatter(x=lengths,y=trend,mode="lines+markers",line=dict(color=CYAN,width=4),marker=dict(size=10,color=PINK),hovertemplate="%{x} seconds: %{y:.1f}%<extra></extra>"))
        trend_fig.add_hline(y=float(bundle["best_t_rf"])*100,line_dash="dash",line_color="#bba8ff",annotation_text="Viral-label cutoff")
        trend_fig.update_layout(template="plotly_dark",height=310,xaxis_title="Video length (seconds)",yaxis_title="Model score (%)",yaxis_range=[0,100],paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color=WHITE,margin=dict(l=10,r=15,t=25,b=20))
        st.plotly_chart(trend_fig,use_container_width=True,config={"displayModeBar":False})
        st.caption("These six example scripts were written for the demo and scored by our trained model. They are not a random or representative sample of the dataset. Differences are associations in model predictions, not proven causes of virality.")
        with st.expander("View all model-generated comparison results"):
            st.dataframe(dashboard_df,hide_index=True,use_container_width=True)
    except Exception as exc:
        st.error(f"The dashboard could not generate predictions: {exc}")

elif page == "Predict virality":
    hero("Live prediction studio", 'Will your video go <span class="vl-pink">viral?</span>', "Describe your TikTok video, and see what our model predicts.")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("### 01 / Create your video")
        examples = {
            "Choose an example…": "",
            "Space discovery": "Scientists discovered an unusual planet orbiting a distant star, and its atmosphere may hold clues about life.",
            "Sports moment": "An unbelievable final-minute goal shocked the stadium and changed the result of the championship match.",
            "Cooking trick": "Here is a simple kitchen trick that makes crispy potatoes with just a few ingredients.",
            "Tech update": "A new artificial intelligence tool can summarize long articles and help students study faster.",
        }
        preset = st.selectbox("Try a ready-made idea", list(examples), key="predict_preset")
        if preset != "Choose an example…" and st.session_state.get("last_preset") != preset:
            st.session_state["video_text"] = examples[preset]
            st.session_state["last_preset"] = preset
        text = st.text_area("Video transcript / spoken content", height=165, placeholder="Describe what is said in your TikTok video…", key="video_text")
        duration = st.slider("Video duration (seconds)", 1, 60, 25, key="video_duration")
        verified = st.selectbox("Is the TikTok account verified?", ["not verified", "verified"], key="video_verified")
        if st.button("ANALYZE MY VIDEO", type="primary", use_container_width=True):
            try:
                st.session_state["predict_result"] = predict(text, duration, verified)
                st.session_state["predict_signature"] = (text, duration, verified)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    with right:
        card("PROFILE", "Your video details", "Our model considers the words in the video, its duration and whether the account has a verification badge.")
        st.markdown(f'<span class="vl-pill">{len(text.split())} words</span><span class="vl-pill">{duration}s video</span><span class="vl-pill">{html.escape(verified)}</span>', unsafe_allow_html=True)
        st.markdown("### What happens next?")
        st.write("We identify the main theme and tone of your video, compare it with patterns from our research, and estimate its viral potential.")
    if "predict_result" in st.session_state:
        if st.session_state.get("predict_signature") != (text, duration, verified):
            st.warning("Inputs have changed since the last prediction. Click Analyze again for updated results.")
        else:
            st.divider()
            render_result(st.session_state["predict_result"])

elif page == "What-if lab":
    hero("Experiment zone", 'Change the inputs. <span class="vl-cyan">Watch the model.</span>', "Run controlled comparisons to see how the model's output varies. A change in prediction does not prove causation.")
    text = st.text_area("What the video says", value="A surprising discovery about how the human brain works and learns new skills.", height=100, key="lab_text")
    a, b = st.columns(2)
    with a: verified = st.selectbox("Is the TikTok account verified?", ["not verified", "verified"], key="lab_verified")
    with b: selected_duration = st.slider("Highlight duration", 1, 60, 25, key="lab_duration")
    durations = [5, 10, 15, 20, 30, 40, 45, 50, 60]
    if selected_duration not in durations:
        durations = sorted(durations + [selected_duration])
    if st.button("RUN LIVE EXPERIMENT", type="primary", use_container_width=True):
        try:
            rows = []
            with st.spinner("Running our trained model across different durations…"):
                for d in durations:
                    r = predict(text, d, verified)
                    rows.append({"Duration (seconds)": d, "Viral score (%)": round(r["prob"] * 100, 2), "Prediction": r["label"], "Cluster": r["cluster"]})
            st.session_state["lab_df"] = pd.DataFrame(rows)
            st.session_state["lab_signature"] = (text, verified, selected_duration)
        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
    if "lab_df" in st.session_state:
        if st.session_state.get("lab_signature") != (text, verified, selected_duration):
            st.warning("Inputs changed. Run the experiment again to update the chart.")
        else:
            df = st.session_state["lab_df"]
            threshold = float(bundle["best_t_rf"]) * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Duration (seconds)"], y=df["Viral score (%)"], mode="lines+markers", line=dict(color=PINK, width=4), marker=dict(size=11, color=CYAN, line=dict(color="#111827", width=2)), fill="tozeroy", fillcolor="rgba(255,52,123,.07)", name="Model score", hovertemplate="%{x} sec • %{y:.2f}%<extra></extra>"))
            fig.add_hline(y=threshold, line_color=CYAN, line_dash="dash", annotation_text="Decision threshold", annotation_font_color=CYAN)
            fig.add_vline(x=selected_duration, line_color="#b39aff", line_dash="dot")
            fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=WHITE, margin=dict(l=20,r=20,t=20,b=20), xaxis_title="Video duration (seconds)", yaxis_title="Model viral score (%)", yaxis_range=[0,100], hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            c1,c2,c3=st.columns(3)
            c1.metric("Highest model score", f'{df["Viral score (%)"].max():.1f}%')
            c2.metric("Lowest model score", f'{df["Viral score (%)"].min():.1f}%')
            c3.metric("Duration tests", len(df))
            with st.expander("See the experiment results"):
                st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption("The text and creator status are held constant. Duration can also change the assigned cluster. This demonstrates model sensitivity, not a causal effect.")
    else:
        card("READY", "Your experiment awaits", "Press Run Live Experiment to compare model scores across durations and inspect the decision boundary.")

elif page == "Model explorer":
    hero("How our model works", 'Behind the <span class="vl-cyan">prediction.</span>', "See what our model uses, explore the themes it recognizes, and test how it groups video ideas.")
    a,b,c=st.columns(3)
    a.metric("Content themes", int(bundle["nmf_model"].n_components))
    b.metric("Video pattern groups", int(bundle["kproto"].n_clusters))
    c.metric("Prediction cutoff", f'{float(bundle["best_t_rf"]):.0%}')
    st.markdown("### Our five-step process")
    cols=st.columns(5)
    for col,step in zip(cols,["Read the video text", "Recognize its theme", "Check its tone", "Match video patterns", "Make a prediction"]):
        with col: st.markdown(f'<div class="vl-step">{step}</div>', unsafe_allow_html=True)
    st.markdown("### Which signals does our model use most?")
    st.caption("Longer bars mean the model relied more on that signal when making decisions. They do not mean changing it will cause a video to go viral.")
    try:
        names=bundle["pipe_preprocess"].get_feature_names_out()
        values=bundle["pipe_model"].feature_importances_
        def readable_feature(name):
            n = str(name).lower()
            if "word_count" in n: return "How much is said"
            if "sentiment" in n: return "Emotional tone"
            if "verified" in n: return "Account verification" + (" — verified" if "not verified" not in n and "verified" in n else "")
            if "duration" in n: return "Video length: " + str(name).split("_")[-1]
            if "topic" in n: return "Video theme: " + str(name).split("_")[-1]
            if "cluster" in n: return "Video pattern group: " + str(name).split("_")[-1]
            return str(name).replace("num__", "").replace("cat__", "").replace("_", " ")
        imp=pd.DataFrame({"Feature":names,"Importance":values}).nlargest(10,"Importance").sort_values("Importance")
        imp["Friendly"] = imp["Feature"].map(readable_feature)
        fig=go.Figure(go.Bar(x=imp["Importance"],y=imp["Friendly"],orientation="h",marker=dict(color=imp["Importance"],colorscale=[[0,"#fe2c55"],[.5,"#ff347b"],[1,"#25f4ee"]],showscale=False),hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>"))
        fig.update_layout(template="plotly_dark",height=480,margin=dict(l=10,r=20,t=20,b=25),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color=WHITE,xaxis_title="Relative influence within the model (not a causal effect)")
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.info(f"Feature importance is unavailable for this exported pipeline: {exc}")
    st.markdown("### Find your video’s pattern group")
    st.write("Try an example and see which group of similar patterns our model assigns it to. Group numbers are identifiers, not quality ratings.")
    probe_text=st.text_input("What does your video say?", "A new invention could change how students learn and study.", key="probe_text")
    x,y=st.columns(2)
    with x: probe_duration=st.slider("Duration",1,60,20,key="probe_duration")
    with y: probe_verified=st.selectbox("Is the TikTok account verified?",["not verified","verified"],key="probe_verified")
    if st.button("FIND MY VIDEO GROUP",type="primary",use_container_width=True):
        try:
            r=predict(probe_text,probe_duration,probe_verified)
            st.success(f'Assigned to **Cluster {r["cluster"]}** · Topic: **{r["topic"]}** · Sentiment: **{r["sentiment"]:+.2f}**')
        except Exception as exc: st.error(f"Cluster inspection failed: {exc}")
    with st.expander("Explore the words behind each theme"):
        try:
            words=bundle["tfidf_vectorizer"].get_feature_names_out()
            components=bundle["nmf_model"].components_
            for i,weights in enumerate(components):
                top=np.argsort(weights)[-8:][::-1]
                st.markdown(f'**Topic {i}:** '+" · ".join(str(words[j]) for j in top))
        except Exception as exc: st.info(f"Topic keywords unavailable: {exc}")
    st.caption("Feature importance and cluster assignments are descriptive properties of the fitted models. They do not establish causal drivers of virality.")

elif page == "Challenge the model":
    hero("Interactive game", 'Can you <span class="vl-pink">beat the model?</span>', "Build two competing TikTok concepts. Guess which receives the higher viral score, then reveal the trained model's verdict.")
    if "game_rounds" not in st.session_state: st.session_state["game_rounds"]=0
    if "game_correct" not in st.session_state: st.session_state["game_correct"]=0
    s1,s2,s3=st.columns(3)
    s1.metric("Rounds played",st.session_state["game_rounds"])
    s2.metric("Correct guesses",st.session_state["game_correct"])
    s3.metric("Your score",f'{st.session_state["game_correct"]/max(1,st.session_state["game_rounds"]):.0%}')
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("### VIDEO A")
        t1=st.text_area("Video A transcript","A fascinating discovery about the solar system and distant planets.",height=130,key="game_a_text")
        d1=st.slider("Video A duration",1,60,15,key="game_a_duration")
        v1=st.selectbox("Is account A verified?",["not verified","verified"],key="game_a_verified")
    with b:
        st.markdown("### VIDEO B")
        t2=st.text_area("Video B transcript","An incredible sports match broke a world record today.",height=130,key="game_b_text")
        d2=st.slider("Video B duration",1,60,45,key="game_b_duration")
        v2=st.selectbox("Is account B verified?",["not verified","verified"],key="game_b_verified")
    guess=st.radio("🎯 Your guess — which gets the higher model score?",["Video A","Video B"],horizontal=True)
    if st.button("REVEAL THE MODEL VERDICT",type="primary",use_container_width=True):
        try:
            r1=predict(t1,d1,v1);r2=predict(t2,d2,v2)
            winner="Tie" if abs(r1["prob"]-r2["prob"])<1e-9 else ("Video A" if r1["prob"]>r2["prob"] else "Video B")
            correct=(winner==guess)
            st.session_state["game_rounds"]+=1
            if correct: st.session_state["game_correct"]+=1
            st.session_state["game_result"]=(r1,r2,winner,correct)
            st.session_state["game_signature"]=(t1,d1,v1,t2,d2,v2,guess)
            st.rerun()
        except Exception as exc: st.error(f"Game prediction failed: {exc}")
    if "game_result" in st.session_state:
        if st.session_state.get("game_signature")!=(t1,d1,v1,t2,d2,v2,guess):
            st.warning("You've changed the videos or your guess. Reveal again to start a new round.")
        else:
            r1,r2,winner,correct=st.session_state["game_result"]
            st.markdown(f'<div class="vl-result"><div class="vl-eyebrow">THE VERDICT</div><h2>{"You guessed the model’s winner!" if correct else ("The model disagrees with your guess" if winner!="Tie" else "The model scored them equally")}</h2><p>{"Higher model score: "+winner if winner!="Tie" else "Both videos received the same model score"}. This does not establish which video would actually perform better.</p></div>',unsafe_allow_html=True)
            x,y=st.columns(2)
            with x:
                st.metric("Video A", f'{r1["prob"]:.1%}')
                st.markdown(f'<div class="vl-cluster-chip">Cluster {r1["cluster"]}</div>', unsafe_allow_html=True)
            with y:
                st.metric("Video B", f'{r2["prob"]:.1%}')
                st.markdown(f'<div class="vl-cluster-chip">Cluster {r2["cluster"]}</div>', unsafe_allow_html=True)
            fig=go.Figure(go.Bar(x=["Video A","Video B"],y=[r1["prob"]*100,r2["prob"]*100],marker_color=[PINK,CYAN],text=[f'{r1["prob"]:.1%}',f'{r2["prob"]:.1%}'],textposition="outside"))
            fig.add_hline(y=float(bundle["best_t_rf"])*100,line_dash="dash",line_color="#bba8ff",annotation_text="Decision threshold")
            fig.update_layout(template="plotly_dark",height=310,yaxis_range=[0,110],yaxis_title="Model viral score (%)",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color=WHITE,margin=dict(l=10,r=10,t=20,b=10),showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Scores are predictions from the trained model, not actual TikTok results. Repeated attempts on the same examples can improve game score without measuring predictive accuracy.")


# ─────────────────────────────────────
# VIRALLAB FOOTER
# ─────────────────────────────────────

import textwrap

st.markdown(textwrap.dedent("""
<style>
.virallab-footer {
    margin-top: 4rem;
    padding: 28px 0 18px;
    border-top: 1px solid #293544;
    color: #aebbd0;
    font-size: 13px;
}

.virallab-footer-heading {
    color: #25F4EE;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}

.virallab-team-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px 20px;
    margin: 18px 0 24px;
}

.virallab-member {
    color: #ffffff;
    font-size: 14px;
}

.virallab-member::before {
    content: "●";
    color: #FE2C55;
    margin-right: 10px;
}

.virallab-footer-bottom {
    border-top: 1px solid #293544;
    padding-top: 20px;
    margin-top: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.virallab-developer-credit {
    text-align: right;
}

.virallab-developer-label {
    color: #aebbd0;
    font-size: 12px;
    margin-bottom: 5px;
}

.virallab-footer a {
    color: #25F4EE !important;
    text-decoration: none;
    font-weight: 600;
    font-size: 13px;
}

.virallab-footer a:hover {
    color: #FE2C55 !important;
}

.virallab-copyright {
    color: #8794a8;
    font-size: 11px;
}

@media (max-width: 480px) {
    .virallab-team-grid {
        grid-template-columns: 1fr;
    }

    .virallab-developer-credit {
        text-align: left;
    }
}
</style>

<div class="virallab-footer">
<div class="virallab-footer-heading">THE TEAM BEHIND VIRALLAB</div>
<div>A collaborative data mining project by</div>

<div class="virallab-team-grid">
<div class="virallab-member">Walh Altayyar</div>
<div class="virallab-member">Latifa Alrasheed</div>
<div class="virallab-member">Reelam Alasmari</div>
<div class="virallab-member">Aljohara Alhazzani</div>
<div class="virallab-member">Randah Alrefay</div>
<div class="virallab-member">Fatimah Babtain</div>
</div>

<div class="virallab-footer-bottom">
<span class="virallab-copyright">
© 2026 ViralLab. All rights reserved.
</span>

<div class="virallab-developer-credit">
<div class="virallab-developer-label">
Website designed &amp; developed by
</div>
<a href="https://github.com/walh"
target="_blank"
rel="noopener noreferrer">
GitHub · @walhaltayyar
</a>
</div>
</div>
</div>
"""), unsafe_allow_html=True)

