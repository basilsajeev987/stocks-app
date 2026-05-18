from transformers import pipeline

# =========================
# LOAD FINBERT
# =========================

classifier = pipeline(

    "text-classification",

    model=
    "ProsusAI/finbert"
)

# =========================
# ANALYZE SENTIMENT
# =========================

def analyze_finbert_sentiment(
    text
):

    try:

        result = classifier(text)[0]

        label = result[
            "label"
        ]

        score = float(
            result["score"]
        )

        sentiment = "Neutral"

        if label.lower() == "positive":

            sentiment = "Bullish"

        elif label.lower() == "negative":

            sentiment = "Bearish"

        return {

            "sentiment":
                sentiment,

            "confidence":
                round(
                    score * 100,
                    2
                ),

            "rawLabel":
                label,
        }

    except Exception as e:

        print(
            "FINBERT ERROR:",
            e
        )

        return {
            "error":
                str(e)
        }