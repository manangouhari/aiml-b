def infer_sentiment(score):
    if score >= 0.5:
        return "Extremely Positive"

    elif score >= 0.05:
        return "Positive"

    elif score <= -0.05 and score > -0.5:
        return "Negative"

    elif score <= -0.5:
        return "Extremely Negative"

    else:
        return "Neutral"
