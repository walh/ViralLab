# ViralLab — interactive TikTok data-mining demo

## Run locally

1. Extract this ZIP.
2. Open a terminal in the `ViralLab` folder.
3. Install requirements: `python -m pip install -r requirements.txt`
4. Launch: `python -m streamlit run app.py`

The app loads the `virallab_models.joblib` exported from your Colab notebook. This is a Python pickle-based format; load **only files you personally created or trust**. Model package versions may need to match your Colab environment for compatibility.

## Included experiences
- Real model prediction with topic, cluster, probability, and decision threshold.
- What-if experiment varying duration while holding other inputs fixed.
- Model pipeline explanation and Random Forest feature importance.
- Two-video challenge game comparing model predictions.

## Academic interpretation
This is an experimental classifier, not a reliable forecast of real TikTok success. It does not use engagement counts as prediction inputs. The challenge compares model scores, not observed outcomes. Feature importance and slider changes do not establish causation.
