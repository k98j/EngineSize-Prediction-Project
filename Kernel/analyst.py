import os

from dotenv import load_dotenv

load_dotenv()


def _fallback_dashboard_analysis(context, question):
    return {
        "executive_summary": (
            "Kernel reviewed the selected tickers using recent market data, return, volatility, "
            "drawdown, moving-average trend, valuation, and portfolio concentration metrics."
        ),
        "market_read": (
            "The strongest names are usually those with positive one-year return, price above the "
            "200-day moving average, and controlled drawdown. Weakness appears when volatility and "
            "drawdown rise faster than return."
        ),
        "portfolio_view": (
            "The current portfolio should be evaluated through weighted return, annualized volatility, "
            "Sharpe ratio, and ticker concentration. A diversified allocation usually avoids depending "
            "too heavily on one high-volatility name."
        ),
        "recommendations": (
            "Compare your current weights with the suggested inverse-volatility weights. If a ticker has "
            "high volatility and weak trend, reduce concentration before adding exposure."
        ),
        "caveats": (
            f"Question considered: {question}. This fallback analysis is rule-based because no valid "
            "OpenAI API key was found. It is educational and not financial advice."
        ),
        "mode": "Rule-based fallback",
    }


def analyze_dashboard(context, question):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_dashboard_analysis(context, question)

    try:
        import dspy

        model_name = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        lm = dspy.LM(model_name, api_key=api_key)
        dspy.configure(lm=lm)

        class KernelDashboardAnalyst(dspy.Signature):
            """
            Generate a structured educational financial analysis from factual market and portfolio metrics.
            Do not provide personalized financial advice.
            Do not give direct buy, sell, or hold instructions.
            Make claims only from the supplied context.
            """

            context: str = dspy.InputField()
            user_question: str = dspy.InputField()

            executive_summary: str = dspy.OutputField(desc="Short dashboard-level summary.")
            market_read: str = dspy.OutputField(desc="Market trend and ticker comparison.")
            portfolio_view: str = dspy.OutputField(desc="Portfolio risk, return, and concentration comments.")
            recommendations: str = dspy.OutputField(desc="Educational next-step checks, not investment advice.")
            caveats: str = dspy.OutputField(desc="Limitations and risk disclaimers.")

        analyst = dspy.ChainOfThought(KernelDashboardAnalyst)
        response = analyst(context=context, user_question=question)

        return {
            "executive_summary": response.executive_summary,
            "market_read": response.market_read,
            "portfolio_view": response.portfolio_view,
            "recommendations": response.recommendations,
            "caveats": response.caveats,
            "mode": f"DSPy + {model_name}",
        }
    except Exception as exc:
        result = _fallback_dashboard_analysis(context, question)
        result["mode"] = f"Fallback after AI error: {exc}"
        return result
