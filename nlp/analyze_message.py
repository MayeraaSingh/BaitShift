from transformers import pipeline

# Load models once
print("Loading sentiment classifier...")
sentiment = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

print("Loading GPT2 generator...")
generator = pipeline("text-generation", model="gpt2")

# Category keywords in priority order
def get_negative_category(message: str) -> str:
    message = message.lower()

    categories = {
        "Banking/Credentials": ["bank", "account", "password", "otp", "ifsc", "debit", "credit", "upi", "pan"],
        "NSFW/Sexual": ["nude", "sexy", "boobs", "pic", "hot", "alone", "video call", "touch", "strip", "send nudes"],
        "Emotional/Manipulation": ["trust", "secret", "don't tell", "dont tell", "urgent", "emergency", "now", "please"],
        "Threats/Violence": ["kill", "hurt", "beat", "attack", "punish", "murder", "stab", "shoot", "threaten", "harm"],
        "Substance/Illegal": ["weed", "drugs", "vape", "high", "drink", "smoke"],
        "Self-Harm/Despair": ["suicide", "end it", "worthless", "hate myself", "no one cares"],
        "Location/Personal Info": ["address", "live", "school", "home", "city", "pincode", "flat", "apartment"],
        "General Curiosity": ["hey", "name", "age", "how old", "baby", "cute", "school?", "what's up", "wanna chat"],
    }

    for category, keywords in categories.items():
        for word in keywords:
            if word in message:
                return category

    return "Unknown"

# Risk level mapping
category_risk_map = {
    "Banking/Credentials": "High",
    "NSFW/Sexual": "High",
    "Emotional/Manipulation": "High",
    "Threats/Violence": "High",
    "Substance/Illegal": "High",
    "Self-Harm/Despair": "Medium",
    "Location/Personal Info": "Medium",
    "General Curiosity": "Medium",
    "Unknown": "Medium",
}

# Prompt descriptions
common_prompt = "You are a safety assistant. Respond with one short, appropriate line. "

category_prompts = {
    "Banking/Credentials": "Someone is asking for banking details. Say you can't discuss that here and suggest a private channel.",
    "NSFW/Sexual": "Someone is being inappropriate. Politely refuse and suggest talking elsewhere.",
    "Emotional/Manipulation": "Someone is trying to build trust or create urgency. Be cautious and suggest continuing privately.",
    "Threats/Violence": "Someone is making threats. Do not engage and end the conversation.",
    "Substance/Illegal": "Someone is discussing illegal activities. Say it's not safe to chat here.",
    "Self-Harm/Despair": "Someone might be in distress. Be supportive and ask them to talk privately or get help.",
    "Location/Personal Info": "Someone is asking for personal location. Don't reveal anything and redirect to another app.",
    "General Curiosity": "Someone is asking casual questions. Answer vaguely and politely.",
    "Unknown": "Be vague and cautious in your response.",
    "Low": "This seems harmless. Respond in a friendly but vague way."
}

# User instructions based on risk level
def get_user_instructions(risk_level: str) -> str:
    instructions = {
        "Low": "This text is harmless. You can continue chatting.",
        "Medium": "Hmm I'm not sure about this message, copy this reply so I can analyse the next response.",
        "High": "This is a high level risk, I have redirected to a fake chatbot. If attack continues convo, please proceed with caution."
    }
    return instructions.get(risk_level, "Please be cautious while chatting.")

# Main callable function
def analyze_message(message: str) -> dict:
    tone = sentiment(message)[0]['label']

    if tone == "NEGATIVE":
        category = get_negative_category(message)
        risk = category_risk_map[category]
    else:
        category = "Low"
        risk = "Low"

    prompt = f"{common_prompt}{category_prompts[category]}\n\nMessage: {message}\nResponse:"
    output = generator(prompt, max_new_tokens=30, do_sample=True, temperature=0.7, pad_token_id=50256)[0]['generated_text']
    reply = output.split("Response:")[-1].strip()

    # Clean up the reply
    if '.' in reply:
        reply = reply.split('.')[0] + '.'
    elif '\n' in reply:
        reply = reply.split('\n')[0]

    reply = reply.replace(message, "").strip()
    if reply.startswith(',') or reply.startswith(':'):
        reply = reply[1:].strip()

    if risk == "High":
        reply += " http://localhost:8000/securechat"

    # Get user instructions based on risk level
    user_instructions = get_user_instructions(risk)

    return {
        "category": category,
        "risk_level": risk,
        "reply": reply,
        "user_instructions": user_instructions
    }

# Optional test
if __name__ == "__main__":
    user_input = input("Enter suspicious message: ")
    result = analyze_message(user_input)
    print("\n--- ANALYSIS ---")
    print("Category:", result['category'])
    print("Risk Level:", result['risk_level'])
    print("Reply:", result['reply'])
    print("Instructions:", result['user_instructions'])
