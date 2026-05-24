from functools import lru_cache
from io import BytesIO

import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, send_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from analyst import analyze_dashboard


app = Flask(__name__)


DEFAULT_TICKERS = "AAPL, MSFT, GOOGL"
DEFAULT_WEIGHTS = "40, 35, 25"
DEFAULT_QUESTION = "Give me a beginner-friendly portfolio and stock analysis."


def format_money(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_ratio(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def format_market_cap(value):
    if value is None or pd.isna(value):
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def parse_tickers(raw):
    tickers = [item.strip().upper() for item in raw.replace(";", ",").split(",")]
    tickers = [ticker for ticker in tickers if ticker]
    if not tickers:
        raise ValueError("Enter at least one ticker.")
    return list(dict.fromkeys(tickers))[:8]


def parse_weights(raw, count):
    if not raw.strip():
        return [1 / count] * count

    values = [float(item.strip()) for item in raw.replace(";", ",").split(",") if item.strip()]
    if len(values) != count:
        raise ValueError("Weights must match the number of tickers, for example: 40, 35, 25.")
    if any(value < 0 for value in values):
        raise ValueError("Weights cannot be negative.")

    total = sum(values)
    if total <= 0:
        raise ValueError("Weights must add up to more than zero.")
    return [value / total for value in values]


@lru_cache(maxsize=64)
def load_history(ticker):
    history = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if history.empty:
        raise ValueError(f"No market data found for {ticker}.")
    return history


@lru_cache(maxsize=64)
def load_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def ticker_snapshot(ticker, weight):
    history = load_history(ticker)
    info = load_info(ticker)
    close = history["Close"].dropna()
    volume = history["Volume"].dropna()

    current_price = float(close.iloc[-1])
    start_price = float(close.iloc[0])
    one_year_return = (current_price - start_price) / start_price
    daily_returns = close.pct_change().dropna()
    volatility = float(daily_returns.std() * (252 ** 0.5))

    sma_50 = float(close.rolling(50).mean().iloc[-1])
    sma_200 = float(close.rolling(200).mean().iloc[-1])
    running_max = close.cummax()
    max_drawdown = float(((close - running_max) / running_max).min())

    latest_volume = float(volume.iloc[-1]) if not volume.empty else None
    avg_volume = float(volume.tail(30).mean()) if not volume.empty else None
    pe_ratio = info.get("trailingPE")
    market_cap = info.get("marketCap")

    trend = "Bullish" if current_price >= sma_200 else "Cautious"
    if current_price >= sma_50 >= sma_200:
        trend = "Strong uptrend"
    elif current_price < sma_50 < sma_200:
        trend = "Downtrend"

    return {
        "ticker": ticker,
        "weight": weight,
        "weight_text": format_percent(weight),
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "current_price": current_price,
        "current_price_text": format_money(current_price),
        "one_year_return": one_year_return,
        "one_year_return_text": format_percent(one_year_return),
        "volatility": volatility,
        "volatility_text": format_percent(volatility),
        "sma_50": sma_50,
        "sma_50_text": format_money(sma_50),
        "sma_200": sma_200,
        "sma_200_text": format_money(sma_200),
        "max_drawdown": max_drawdown,
        "max_drawdown_text": format_percent(max_drawdown),
        "market_cap": market_cap,
        "market_cap_text": format_market_cap(market_cap),
        "pe_ratio": pe_ratio,
        "pe_ratio_text": format_ratio(pe_ratio),
        "latest_volume_text": "N/A" if latest_volume is None else f"{latest_volume:,.0f}",
        "avg_volume_text": "N/A" if avg_volume is None else f"{avg_volume:,.0f}",
        "trend": trend,
    }


def build_price_frame(tickers):
    series = []
    for ticker in tickers:
        close = load_history(ticker)["Close"].rename(ticker)
        series.append(close)
    prices = pd.concat(series, axis=1).dropna()
    if prices.empty:
        raise ValueError("Could not align price histories for the selected tickers.")
    return prices


def portfolio_metrics(tickers, weights):
    prices = build_price_frame(tickers)
    returns = prices.pct_change().dropna()
    weights_series = pd.Series(weights, index=tickers)
    portfolio_returns = returns.mul(weights_series, axis=1).sum(axis=1)

    annual_return = float((1 + portfolio_returns.mean()) ** 252 - 1)
    annual_volatility = float(portfolio_returns.std() * (252 ** 0.5))
    sharpe = None if annual_volatility == 0 else annual_return / annual_volatility

    equity_curve = (1 + portfolio_returns).cumprod()
    max_drawdown = float(((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min())
    concentration = float(sum(weight ** 2 for weight in weights))
    correlation = returns.corr().round(2)

    vol_by_ticker = returns.std() * (252 ** 0.5)
    inverse_vol = 1 / vol_by_ticker.replace(0, float("nan"))
    suggested = (inverse_vol / inverse_vol.sum()).fillna(1 / len(tickers))

    momentum = prices.iloc[-1] / prices.iloc[0] - 1
    momentum_score = momentum.clip(lower=0.01) / vol_by_ticker.replace(0, float("nan"))
    momentum_weights = (momentum_score / momentum_score.sum()).fillna(1 / len(tickers))

    return {
        "annual_return": annual_return,
        "annual_return_text": format_percent(annual_return),
        "annual_volatility": annual_volatility,
        "annual_volatility_text": format_percent(annual_volatility),
        "sharpe": sharpe,
        "sharpe_text": format_ratio(sharpe),
        "max_drawdown": max_drawdown,
        "max_drawdown_text": format_percent(max_drawdown),
        "concentration": concentration,
        "concentration_text": format_ratio(concentration),
        "correlation": correlation,
        "suggested_weights": [
            {"ticker": ticker, "weight_text": format_percent(float(suggested[ticker]))}
            for ticker in tickers
        ],
        "momentum_weights": [
            {"ticker": ticker, "weight_text": format_percent(float(momentum_weights[ticker]))}
            for ticker in tickers
        ],
    }


def build_context(snapshots, metrics):
    lines = [
        "KERNEL FINANCIAL DASHBOARD CONTEXT",
        f"Portfolio annualized return: {metrics['annual_return_text']}",
        f"Portfolio annualized volatility: {metrics['annual_volatility_text']}",
        f"Portfolio Sharpe proxy: {metrics['sharpe_text']}",
        f"Portfolio max drawdown: {metrics['max_drawdown_text']}",
        f"Concentration index: {metrics['concentration_text']}",
        "",
        "Ticker rows:",
    ]
    for row in snapshots:
        lines.append(
            f"{row['ticker']}: weight {row['weight_text']}, price {row['current_price_text']}, "
            f"1Y return {row['one_year_return_text']}, volatility {row['volatility_text']}, "
            f"max drawdown {row['max_drawdown_text']}, P/E {row['pe_ratio_text']}, trend {row['trend']}."
        )
    return "\n".join(lines)


def analyze_request(tickers_raw, weights_raw, question):
    tickers = parse_tickers(tickers_raw)
    weights = parse_weights(weights_raw, len(tickers))
    snapshots = [ticker_snapshot(ticker, weight) for ticker, weight in zip(tickers, weights)]
    metrics = portfolio_metrics(tickers, weights)
    context = build_context(snapshots, metrics)
    analysis = analyze_dashboard(context, question)
    return {
        "tickers": tickers,
        "weights_raw": weights_raw,
        "snapshots": snapshots,
        "metrics": metrics,
        "context": context,
        "analysis": analysis,
    }


def make_pdf(result, question):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=8.5, leading=11))
    story = [
        Paragraph("Kernel AI Financial Analyst Report", styles["Title"]),
        Paragraph("Educational report, not financial advice.", styles["Small"]),
        Spacer(1, 12),
        Paragraph("Question", styles["Heading2"]),
        Paragraph(question, styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Portfolio Metrics", styles["Heading2"]),
    ]
    metrics = result["metrics"]
    metric_rows = [
        ["Annualized Return", metrics["annual_return_text"]],
        ["Annualized Volatility", metrics["annual_volatility_text"]],
        ["Sharpe Proxy", metrics["sharpe_text"]],
        ["Max Drawdown", metrics["max_drawdown_text"]],
        ["Concentration Index", metrics["concentration_text"]],
    ]
    metric_table = Table(metric_rows, colWidths=[8 * cm, 7 * cm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7FB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([metric_table, Spacer(1, 10), Paragraph("Ticker Snapshot", styles["Heading2"])])

    rows = [["Ticker", "Weight", "Price", "1Y Return", "Volatility", "Trend"]]
    for row in result["snapshots"]:
        rows.append([
            row["ticker"],
            row["weight_text"],
            row["current_price_text"],
            row["one_year_return_text"],
            row["volatility_text"],
            row["trend"],
        ])
    table = Table(rows, colWidths=[2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.5 * cm, 3.2 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17202A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 12)])

    for title, key in [
        ("Executive Summary", "executive_summary"),
        ("Market Read", "market_read"),
        ("Portfolio View", "portfolio_view"),
        ("Educational Recommendations", "recommendations"),
        ("Caveats", "caveats"),
    ]:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Paragraph(str(result["analysis"][key]).replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer


@app.route("/", methods=["GET", "POST"])
def index():
    tickers = DEFAULT_TICKERS
    weights = DEFAULT_WEIGHTS
    question = DEFAULT_QUESTION
    result = None
    error = None

    if request.method == "POST":
        tickers = request.form.get("tickers", tickers).strip()
        weights = request.form.get("weights", weights).strip()
        question = request.form.get("question", question).strip()
        try:
            result = analyze_request(tickers, weights, question)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        tickers=tickers,
        weights=weights,
        question=question,
        result=result,
        error=error,
    )


@app.route("/report", methods=["POST"])
def report():
    tickers = request.form.get("tickers", DEFAULT_TICKERS).strip()
    weights = request.form.get("weights", DEFAULT_WEIGHTS).strip()
    question = request.form.get("question", DEFAULT_QUESTION).strip()
    result = analyze_request(tickers, weights, question)
    pdf = make_pdf(result, question)
    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="kernel_financial_report.pdf",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
