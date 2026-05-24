# Kernel

Kernel is an AI financial analyst dashboard built with Flask, DSPy, OpenAI, yfinance, pandas, and ReportLab.

## What It Does

- Retrieves recent public stock-market data with `yfinance`
- Computes per-ticker metrics: price, 1-year return, annualized volatility, moving averages, drawdown, P/E, market cap, and volume proxy
- Computes portfolio metrics: weighted return, annualized volatility, Sharpe proxy, max drawdown, concentration, and correlation matrix
- Generates risk-balanced and momentum-tilted educational weight suggestions
- Answers natural-language portfolio questions through a DSPy reasoning module when `OPENAI_API_KEY` is configured
- Falls back to a rule-based analyst when no API key is present
- Exports a PDF financial report

## Project Structure

```text
Kernel/
  app.py
  analyst.py
  requirements.txt
  Procfile
  .env.example
  templates/index.html
  static/style.css
```

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=openai/gpt-4o-mini
```

Do not upload `.env` to GitHub.

## Deploy

GitHub Pages cannot run Flask apps. Deploy Kernel on Render, Railway, or PythonAnywhere.

For Render:

1. Push this folder to GitHub.
2. Create a new Web Service.
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
gunicorn app:app
```

5. Add `OPENAI_API_KEY` in Render environment variables.

## Disclaimer

Kernel is an educational portfolio project, not financial advice.
