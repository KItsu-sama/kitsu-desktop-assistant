import time

BASE_SCORE = {
    "SHORT": 0.2,
    "EPISODIC": 0.5,
    "LONG": 0.8
}

EMOTION_BOOST = {
    "joy": 0.15,
    "anger": 0.25,
    "love": 0.15,
    "fear": 0.2,
    "satisfy": 0.1,    
    "sadness": 0.2,
    "neutral": 0.0
}

DECAY_BY_TYPE = {
    "SHORT": 0.08,     # fast decay
    "EPISODIC": 0.03,  # slower
    "LONG": 0.005      # very slow
}

def compute_score(mem, now, current_emotion=None):
    score = BASE_SCORE.get(mem.get("type", "SHORT"), 0.2)

    # Emotion amplifier (see section 2)
    emotion = mem.get("emotion", "neutral")
    score += EMOTION_BOOST.get(emotion, 0.0)

    if emotion in ("love", "fear", "anger"):
        score += 0.1      # emotional inertia

    if current_emotion and emotion != current_emotion:
        score -= 0.05     # emotion mismatch decay

    # Recency
    age_hours = (now - mem["timestamp"]) / 3600
    score += 0.2 if age_hours < 1 else 0.0

    # Reinforcement
    score += min(0.4, mem.get("uses", 0) * 0.06)

    # 🧠 True decay
    decay_rate = DECAY_BY_TYPE.get(mem.get("type"), 0.05)
    score -= age_hours * decay_rate


    return round(max(0.0, min(score, 1.0)), 3)
