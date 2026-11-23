"""ViralLab — TikTok-inspired interactive showcase for the original trained model."""
from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="ViralLab • TikTok Intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
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
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading your trained TikTok models…")
def load_bundle():
    # Load only the model file exported from your own trusted notebook.
    return joblib.load(Path(__file__).with_name("virallab_models.joblib"))

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Your model bundle could not be loaded: {exc}")
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
    st.markdown(f'<div class="vl-result"><div class="vl-eyebrow">⚡ ANALYSIS COMPLETE</div><h2>{"🚀 Predicted VIRAL" if label == "Viral" else "🌙 Predicted NON-VIRAL"}</h2><p>This is your trained model’s classification, not a guarantee of real-world performance.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model viral score", f'{result["prob"]:.1%}')
    c2.metric("Decision threshold", f'{result["threshold"]:.1%}')
    c3.metric("Assigned cluster", f'#{result["cluster"]}')
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        probability_chart(result["prob"], result["threshold"])
    with right:
        st.markdown("### 🧬 What the model found")
        st.markdown(f'<span class="vl-pill">Topic: {html.escape(result["topic"])}</span><span class="vl-pill">Duration: {result["duration"]}</span><span class="vl-pill">Sentiment: {result["sentiment"]:+.2f}</span>', unsafe_allow_html=True)
        st.write("The cyan marker is the model's decision threshold. Scores above that threshold receive the **Viral** label.")
        st.caption("This score is a model estimate, not a calibrated guarantee of TikTok reach.")

st.sidebar.markdown('<div class="vl-brand">⚡ Viral<span class="vl-pink">Lab</span></div>', unsafe_allow_html=True)
st.sidebar.caption("TIKTOK INTELLIGENCE • DATA MINING")
st.sidebar.markdown('<div class="vl-divider"></div>', unsafe_allow_html=True)
page = st.sidebar.radio("EXPLORE", ["🏠 Home", "🎯 Predict virality", "🎛️ What-if lab", "🧬 Model explorer", "🎮 Challenge the AI"], label_visibility="collapsed")
st.sidebar.markdown('<div class="vl-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<span class="vl-pill">● Model connected</span>', unsafe_allow_html=True)
st.sidebar.caption("Real trained NLP, K-Prototypes and Random Forest components. Experimental predictions only.")

if page == "🏠 Home":
    hero("The TikTok data mining experience", 'Can data predict <span class="vl-pink">viral</span> content?', "Discover how language, creator status and video duration flow through topic modeling, clustering and a trained classification model.")
    a, b, c = st.columns(3)
    with a: card("🧠", "Real machine learning", "Your original TF-IDF, NMF, VADER, K-Prototypes and Random Forest pipeline.")
    with b: card("🎛️", "Live experiments", "Change one variable and watch the trained model respond in real time.")
    with c: card("🏆", "Challenge the AI", "Compare two video ideas and try to anticipate the model's ranking.")
    st.markdown("### ⚡ Your model at a glance")
    a, b, c, d = st.columns(4)
    a.metric("Cleaned records", "19,084", help="Count reported in the original notebook, not dynamically reloaded.")
    b.metric("NMF topics", int(bundle["nmf_model"].n_components))
    c.metric("K-Prototypes clusters", int(bundle["kproto"].n_clusters))
    d.metric("Decision threshold", f'{float(bundle["best_t_rf"]):.2f}')
    st.markdown("### 🔍 How the prediction happens")
    cols = st.columns(5)
    for col, step in zip(cols, ["📝 Transcript", "🔤 TF-IDF + NMF", "💬 Sentiment", "🧬 K-Prototypes", "🎯 Random Forest"]):
        with col: st.markdown(f'<div class="vl-step">{step}</div>', unsafe_allow_html=True)
    st.info("Academic integrity note: the model does not use video views, likes or shares as prediction inputs. Its performance is limited; this is an experimental learning tool.")
    if st.button("🎯 Start a live prediction", type="primary", use_container_width=True):
        st.session_state["home_go"] = True
    if st.session_state.get("home_go"):
        st.success("Select **🎯 Predict virality** in the sidebar to begin.")

elif page == "🎯 Predict virality":
    hero("Live prediction studio", 'Will your video go <span class="vl-pink">viral?</span>', "Write a TikTok idea, choose its duration and creator status, then reveal your real model's classification.")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("### 01 / Create your video")
        examples = {
            "Choose an example…": "",
            "🔭 Space discovery": "Scientists discovered an unusual planet orbiting a distant star, and its atmosphere may hold clues about life.",
            "⚽ Sports moment": "An unbelievable final-minute goal shocked the stadium and changed the result of the championship match.",
            "🍳 Cooking trick": "Here is a simple kitchen trick that makes crispy potatoes with just a few ingredients.",
            "💻 Tech update": "A new artificial intelligence tool can summarize long articles and help students study faster.",
        }
        preset = st.selectbox("Try a ready-made idea", list(examples), key="predict_preset")
        if preset != "Choose an example…" and st.session_state.get("last_preset") != preset:
            st.session_state["video_text"] = examples[preset]
            st.session_state["last_preset"] = preset
        text = st.text_area("Video transcript / spoken content", height=165, placeholder="Describe what is said in your TikTok video…", key="video_text")
        duration = st.slider("⏱️ Video duration (seconds)", 1, 60, 25, key="video_duration")
        verified = st.selectbox("✔️ Creator verification", ["not verified", "verified"], key="video_verified")
        if st.button("⚡ ANALYZE MY VIDEO", type="primary", use_container_width=True):
            try:
                st.session_state["predict_result"] = predict(text, duration, verified)
                st.session_state["predict_signature"] = (text, duration, verified)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    with right:
        card("📱", "Your video profile", "The model analyzes spoken content, length and creator verification — not historical likes or views.")
        st.markdown(f'<span class="vl-pill">{len(text.split())} words</span><span class="vl-pill">{duration}s video</span><span class="vl-pill">{html.escape(verified)}</span>', unsafe_allow_html=True)
        st.markdown("### What happens next?")
        st.write("Your text is transformed into a topic and sentiment score. K-Prototypes assigns a cluster, and Random Forest generates a score compared with the learned threshold.")
    if "predict_result" in st.session_state:
        if st.session_state.get("predict_signature") != (text, duration, verified):
            st.warning("Inputs have changed since the last prediction. Click Analyze again for updated results.")
        else:
            st.divider()
            render_result(st.session_state["predict_result"])

elif page == "🎛️ What-if lab":
    hero("Experiment zone", 'Change the inputs. <span class="vl-cyan">Watch the model.</span>', "Run controlled comparisons to see how the model's output varies. A change in prediction does not prove causation.")
    text = st.text_area("📝 Video transcript", value="A surprising discovery about how the human brain works and learns new skills.", height=100, key="lab_text")
    a, b = st.columns(2)
    with a: verified = st.selectbox("Creator verification", ["not verified", "verified"], key="lab_verified")
    with b: selected_duration = st.slider("Highlight duration", 1, 60, 25, key="lab_duration")
    durations = [5, 10, 15, 20, 30, 40, 45, 50, 60]
    if selected_duration not in durations:
        durations = sorted(durations + [selected_duration])
    if st.button("🧪 RUN LIVE EXPERIMENT", type="primary", use_container_width=True):
        try:
            rows = []
            with st.spinner("Running your trained model across different durations…"):
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
            with st.expander("📋 See the experiment results"):
                st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption("The text and creator status are held constant. Duration can also change the assigned cluster. This demonstrates model sensitivity, not a causal effect.")
    else:
        card("🧪", "Your experiment awaits", "Press Run Live Experiment to compare model scores across durations and inspect the decision boundary.")

elif page == "🧬 Model explorer":
    hero("Under the hood", 'Explore the <span class="vl-cyan">intelligence.</span>', "Inspect the trained pipeline, topic vocabulary, cluster assignments and Random Forest feature importance.")
    a,b,c=st.columns(3)
    a.metric("Topics", int(bundle["nmf_model"].n_components))
    b.metric("Clusters", int(bundle["kproto"].n_clusters))
    c.metric("Classification threshold", f'{float(bundle["best_t_rf"]):.2f}')
    st.markdown("### 🔬 Pipeline architecture")
    cols=st.columns(5)
    for col,step in zip(cols,["📝 Transcript", "🧠 NMF topic", "💬 VADER sentiment", "🧬 K-Prototypes", "🎯 Random Forest"]):
        with col: st.markdown(f'<div class="vl-step">{step}</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Feature importance — actual Random Forest")
    try:
        names=bundle["pipe_preprocess"].get_feature_names_out()
        values=bundle["pipe_model"].feature_importances_
        imp=pd.DataFrame({"Feature":names,"Importance":values}).nlargest(15,"Importance").sort_values("Importance")
        fig=go.Figure(go.Bar(x=imp["Importance"],y=imp["Feature"],orientation="h",marker=dict(color=imp["Importance"],colorscale=[[0,"#5b3b83"],[.5,PINK],[1,CYAN]],showscale=False),hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>"))
        fig.update_layout(template="plotly_dark",height=480,margin=dict(l=10,r=20,t=20,b=25),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color=WHITE,xaxis_title="Relative feature importance")
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.info(f"Feature importance is unavailable for this exported pipeline: {exc}")
    st.markdown("### 🧬 Interactive cluster probe")
    st.write("Submit a sample transcript to see which **trained K-Prototypes cluster** it belongs to. This is a real assignment, not a fabricated cluster scatterplot.")
    probe_text=st.text_input("Sample transcript", "A new invention could change how students learn and study.", key="probe_text")
    x,y=st.columns(2)
    with x: probe_duration=st.slider("Duration",1,60,20,key="probe_duration")
    with y: probe_verified=st.selectbox("Creator",["not verified","verified"],key="probe_verified")
    if st.button("🔍 INSPECT CLUSTER",type="primary",use_container_width=True):
        try:
            r=predict(probe_text,probe_duration,probe_verified)
            st.success(f'Assigned to **Cluster {r["cluster"]}** · Topic: **{r["topic"]}** · Sentiment: **{r["sentiment"]:+.2f}**')
        except Exception as exc: st.error(f"Cluster inspection failed: {exc}")
    with st.expander("🧠 View NMF topic keywords"):
        try:
            words=bundle["tfidf_vectorizer"].get_feature_names_out()
            components=bundle["nmf_model"].components_
            for i,weights in enumerate(components):
                top=np.argsort(weights)[-8:][::-1]
                st.markdown(f'**Topic {i}:** '+" · ".join(str(words[j]) for j in top))
        except Exception as exc: st.info(f"Topic keywords unavailable: {exc}")
    st.caption("Feature importance and cluster assignments are descriptive properties of the fitted models. They do not establish causal drivers of virality.")

else:
    hero("Interactive AI game", 'Can you <span class="vl-pink">beat the model?</span>', "Build two competing TikTok concepts. Guess which receives the higher viral score, then reveal the trained model's verdict.")
    if "game_rounds" not in st.session_state: st.session_state["game_rounds"]=0
    if "game_correct" not in st.session_state: st.session_state["game_correct"]=0
    s1,s2,s3=st.columns(3)
    s1.metric("Rounds played",st.session_state["game_rounds"])
    s2.metric("Correct guesses",st.session_state["game_correct"])
    s3.metric("Your score",f'{st.session_state["game_correct"]/max(1,st.session_state["game_rounds"]):.0%}')
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("### 🩷 VIDEO A")
        t1=st.text_area("Video A transcript","A fascinating discovery about the solar system and distant planets.",height=130,key="game_a_text")
        d1=st.slider("Video A duration",1,60,15,key="game_a_duration")
        v1=st.selectbox("Video A creator",["not verified","verified"],key="game_a_verified")
    with b:
        st.markdown("### 🩵 VIDEO B")
        t2=st.text_area("Video B transcript","An incredible sports match broke a world record today.",height=130,key="game_b_text")
        d2=st.slider("Video B duration",1,60,45,key="game_b_duration")
        v2=st.selectbox("Video B creator",["not verified","verified"],key="game_b_verified")
    guess=st.radio("🎯 Your guess — which gets the higher model score?",["Video A","Video B"],horizontal=True)
    if st.button("🏆 REVEAL THE AI VERDICT",type="primary",use_container_width=True):
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
            st.markdown(f'<div class="vl-result"><div class="vl-eyebrow">🏆 THE VERDICT</div><h2>{"🎉 You guessed the model’s winner!" if correct else ("🤖 The model disagrees with your guess" if winner!="Tie" else "🤝 The model scored them equally")}</h2><p>{"Higher model score: "+winner if winner!="Tie" else "Both videos received the same model score"}. This does not establish which video would actually perform better.</p></div>',unsafe_allow_html=True)
            x,y=st.columns(2)
            x.metric("🩷 Video A",f'{r1["prob"]:.1%}',f'Cluster {r1["cluster"]}',delta_color="off")
            y.metric("🩵 Video B",f'{r2["prob"]:.1%}',f'Cluster {r2["cluster"]}',delta_color="off")
            fig=go.Figure(go.Bar(x=["Video A","Video B"],y=[r1["prob"]*100,r2["prob"]*100],marker_color=[PINK,CYAN],text=[f'{r1["prob"]:.1%}',f'{r2["prob"]:.1%}'],textposition="outside"))
            fig.add_hline(y=float(bundle["best_t_rf"])*100,line_dash="dash",line_color="#bba8ff",annotation_text="Decision threshold")
            fig.update_layout(template="plotly_dark",height=310,yaxis_range=[0,110],yaxis_title="Model viral score (%)",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color=WHITE,margin=dict(l=10,r=10,t=20,b=10),showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Scores are predictions from the trained model, not actual TikTok results. Repeated attempts on the same examples can improve game score without measuring predictive accuracy.")
