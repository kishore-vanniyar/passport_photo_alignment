# Passport Photo Alignment System

Fixed version of the mini-project prototype.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## What it demonstrates

- Face detection
- Eye detection
- Rotation correction
- Scaling
- Translation
- Affine transformation
- Standard 413 x 531 portrait output

If eye detection fails, the application uses a safe face-crop fallback so that the output does not become a blank image.
