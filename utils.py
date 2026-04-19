import pandas as pd

# Load ads
def load_ads():
    return pd.read_csv("ads.csv")

# Load users safely
def load_users():
    try:
        return pd.read_csv("users.csv")
    except:
        return pd.DataFrame(columns=["user_id", "interest", "time_spent"])

# Content-based filtering
def content_based(interest):
    ads = load_ads()
    return ads[ads["category"] == interest]

# Collaborative filtering (simple)
def collaborative(interest):
    users = load_users()
    similar = users[users["interest"] == interest]
    if not similar.empty:
        return similar["interest"].mode()[0]
    return interest

# Hybrid recommendation
def hybrid(prob, interest):
    final_interest = collaborative(interest)
    ads = content_based(final_interest)

    if ads.empty:
        return "No ads available"

    if prob > 0.7:
        return ads.head(2)
    elif prob > 0.4:
        return ads.head(1)
    else:
        return "Low interest → General ads"