# Heart Disease Risk Predictor

A Streamlit web app that predicts heart disease risk using a saved KNN model, scaler, and training column list.

## Files

- `app.py` - Streamlit app
- `knn_heart_model.pkl` - trained KNN model
- `heart_scaler.pkl` - fitted scaler
- `heart_columns.pkl` - expected model columns
- `requirements.txt` - dependencies for local run and Streamlit Cloud

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Deploy on Streamlit Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and the three `.pkl` files.
3. Go to Streamlit Community Cloud.
4. Choose the repository.
5. Set the main file path to `app.py`.
6. Deploy.

## Note

This app is an educational ML demo and not a medical diagnosis tool.
