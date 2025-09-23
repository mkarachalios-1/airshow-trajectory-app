# Airshow Trajectory Envelope (Wind = 0)

Axes: **x** = display line, **y** = toward crowd, **z** = height.  
Event-based physics: velocity-dependent COR at impact, Coulomb friction impulse, and ground slide with kinetic friction (optional aerodynamic drag). Integrates **until rest**.

## Run online (Streamlit Cloud)
1. Fork or upload this repo to your GitHub account.
2. Go to https://share.streamlit.io, click **New app**, and select your repo.
3. Set **Main file** to `streamlit_app.py`. Deploy and share the URL.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
