RESPONSES = {
    "joy": [
        "That’s wonderful to hear 😊 Tell me what made you feel happy today!",
        "I’m glad you’re feeling positive. Want to share more?"
    ],
    "sadness": [
        "I’m really sorry you’re feeling this way. Do you want to talk about what happened?",
        "That sounds heavy. I’m here with you — what’s on your mind?"
    ],
    "anger": [
        "That sounds frustrating. Want to tell me what triggered it?",
        "I can understand that anger. Let’s unpack it together."
    ],
    "anxiety": [
        "That sounds stressful. Let’s take it one step at a time — what’s worrying you most?",
        "I’m here. Try a slow breath: in 4 seconds, hold 2, out 6. What’s the main fear?"
    ],
    "fear": [
        "That sounds scary. You’re not alone — what part feels most threatening?",
        "I hear you. Let’s identify what you can control right now."
    ],
    "calm": [
        "I’m happy you’re feeling calm. Want to reflect on what helped?",
        "That’s good — maintaining calm is powerful. What’s going well?"
    ]
}

def pick_response(emotion: str) -> str:
    options = RESPONSES.get(emotion, ["I’m here with you. Want to say more?"])
    return options[0]
