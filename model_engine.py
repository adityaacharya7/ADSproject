"""
Self-Contained Machine Learning & Inference Engine for ADS Production Deployment.
Handles:
1. Real-time emotion classification & multi-class calibrated probability extraction
2. VADER sentiment polarity scoring & automated support ticket triage
3. Local token-removal sensitivity explanations
4. Statistical data drift monitoring (histogram-based PSI)
5. Robust model unpickling & artifact management
"""

import os
import re
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from text_preprocessing import clean_tweet_text, apply_negation_tagging

# Match the training implementation; its lexicon is bundled with the package.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / "models"


# The artifact stores its class as __main__.EndToEndEmotionPipeline.
class EndToEndEmotionPipeline:
    @property
    def classes_(self):
        # Probability columns follow the classifier, not emotion_classes metadata.
        return self.classifier.classes_

    def _features(self, texts):
        cleaned = [clean_tweet_text(text) for text in texts]
        tagged = [apply_negation_tagging(text) for text in cleaned]
        scores = [sia.polarity_scores(text) for text in cleaned]
        vader = np.array([[s[k] for k in ('compound', 'pos', 'neg', 'neu')]
                          for s in scores])
        return hstack([self.vectorizer.transform(tagged), self.scaler.transform(vader)],
                      format='csr')

    def predict_proba(self, texts):
        return self.classifier.predict_proba(self._features(texts))

    def predict(self, texts):
        return self.classifier.predict(self._features(texts))


class EmotionEngine:
    """Production Inference, XAI, and Drift Engine."""

    EMOTIONS = ["Anger / Frustration", "Joy / Gratitude", "Disappointment / Sadness", "Fear / Anxiety", "Neutral / Inquiry"]

    # Baseline statistical profile for Twitter Customer Support dataset (from Experiment 3 & 4)
    BASELINE_STATS = {
        "mean_length": 85.4,
        "std_length": 42.1,
        "mean_words": 15.2,
        "std_words": 7.6,
        "mean_vader_compound": -0.12,
        "std_vader_compound": 0.48,
        "emotion_distribution": {
            "Anger / Frustration": 0.38,
            "Joy / Gratitude": 0.22,
            "Disappointment / Sadness": 0.16,
            "Fear / Anxiety": 0.08,
            "Neutral / Inquiry": 0.16
        }
    }

    def __init__(self, model_filename: str = "best_emotion_model.joblib"):
        self.model_path = MODELS_DIR / model_filename
        self.pipeline = None
        self.classes = self.EMOTIONS
        self.drift_baseline = json.loads((MODELS_DIR / 'drift_baseline.json').read_text())
        self._load_model()

    def _load_model(self):
        """Loads and deserializes the champion model artifact."""
        if self.model_path.exists():
            try:
                # Streamlit replaces __main__ on reruns, so register at load time.
                setattr(sys.modules['__main__'], 'EndToEndEmotionPipeline', EndToEndEmotionPipeline)
                bundle = joblib.load(self.model_path)
                if hasattr(bundle, "predict"):
                    self.pipeline = bundle
                elif isinstance(bundle, dict):
                    self.pipeline = bundle.get("pipeline", bundle.get("model"))
                if hasattr(self.pipeline, "classes_"):
                    self.classes = [str(c) for c in self.pipeline.classes_]
            except Exception as e:
                print(f"[!] Warning: Model loading encountered exception: {e}")

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Applies negation tagging and cleaning."""
        if not text or not isinstance(text, str):
            return ""
        t = text.lower().strip()
        t = re.sub(r'https?://\S+|www\.\S+', '', t)
        t = re.sub(r'@\w+', '', t)
        # Contraction handling
        t = re.sub(r"can't", "can not", t)
        t = re.sub(r"won't", "will not", t)
        t = re.sub(r"n't", " not", t)
        # Negation linking: link 'not' / 'never' / 'no' to following token
        t = re.sub(r'\b(not|no|never)\s+([a-z]+)', r'\1_\2', t)
        t = re.sub(r'[^a-z0-9_!?\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Calculates VADER sentiment polarity."""
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment_label = "Positive"
        elif compound <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        return {
            "label": sentiment_label,
            "compound": round(compound, 4),
            "pos": round(scores['pos'], 4),
            "neu": round(scores['neu'], 4),
            "neg": round(scores['neg'], 4)
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Performs full inference, probability calibration, and urgency triage."""
        t0 = time.perf_counter()
        clean = self.preprocess_text(text)
        sentiment = self.get_sentiment(text)

        # Rule-based fallback if ML pipeline is unavailable
        probabilities = {}
        primary_emotion = "Neutral / Inquiry"
        confidence = 0.50

        if self.pipeline is not None and hasattr(self.pipeline, "predict_proba"):
            try:
                # Handle pipeline input format
                raw_probs = self.pipeline.predict_proba([text])[0]
                classes = getattr(self.pipeline, "classes_", self.EMOTIONS)
                for c, p in zip(classes, raw_probs):
                    probabilities[str(c)] = round(float(p), 4)
                
                # Pick max
                sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                primary_emotion = sorted_emotions[0][0]
                confidence = sorted_emotions[0][1]
            except Exception:
                logging.exception('Model inference failed; using heuristic fallback')
                probabilities = self._heuristic_probabilities(clean, sentiment)
                primary_emotion = max(probabilities, key=probabilities.get)
                confidence = probabilities[primary_emotion]
        else:
            probabilities = self._heuristic_probabilities(clean, sentiment)
            primary_emotion = max(probabilities, key=probabilities.get)
            confidence = probabilities[primary_emotion]

        # Secondary emotion detection (if 2nd highest > 0.20 and close to top)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        secondary_emotion = None
        is_mixed = False
        if len(sorted_probs) > 1 and sorted_probs[1][1] >= 0.20:
            secondary_emotion = sorted_probs[1][0]
            is_mixed = True

        # Urgency Level Triage Logic
        compound = sentiment["compound"]
        if ("Anger" in primary_emotion or "Fear" in primary_emotion) and compound <= -0.5:
            urgency = "CRITICAL"
            routing = "Instant Escalation: Route to Senior Supervisor Queue (<15 min SLA)"
            badge_color = "#DC2626"  # Red
        elif compound <= -0.25 or "Anger" in primary_emotion or "Sadness" in primary_emotion:
            urgency = "HIGH"
            routing = "Priority Queue: Route to Specialist Agent (<1 hr SLA)"
            badge_color = "#F97316"  # Orange
        elif compound >= 0.3 or "Joy" in primary_emotion:
            urgency = "LOW"
            routing = "Standard Queue: Positive Feedback / Automated Thank You"
            badge_color = "#10B981"  # Green
        else:
            urgency = "MEDIUM"
            routing = "Standard Queue: General Inquiries (<4 hr SLA)"
            badge_color = "#3B82F6"  # Blue

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "text": text,
            "clean_text": clean,
            "primary_emotion": primary_emotion,
            "confidence": round(float(confidence), 4),
            "secondary_emotion": secondary_emotion,
            "is_mixed_emotion": is_mixed,
            "probabilities": probabilities,
            "sentiment": sentiment,
            "urgency": urgency,
            "routing_recommendation": routing,
            "badge_color": badge_color,
            "latency_ms": latency_ms
        }

    def _heuristic_probabilities(self, text: str, sentiment: Dict[str, Any]) -> Dict[str, float]:
        """Provides calibrated heuristic probabilities based on lexicon patterns."""
        t = text.lower()
        comp = sentiment["compound"]
        scores = {
            "Anger / Frustration": 0.1,
            "Joy / Gratitude": 0.1,
            "Disappointment / Sadness": 0.1,
            "Fear / Anxiety": 0.05,
            "Neutral / Inquiry": 0.2
        }
        if any(w in t for w in ["terrible", "worst", "unacceptable", "furious", "delayed", "angry", "horrible", "hate"]):
            scores["Anger / Frustration"] += 0.5
        if any(w in t for w in ["thank", "great", "awesome", "helpful", "love", "resolved", "excellent", "fast"]):
            scores["Joy / Gratitude"] += 0.5
        if any(w in t for w in ["sad", "disappointed", "missed", "lost", "regret", "broken"]):
            scores["Disappointment / Sadness"] += 0.4
        if any(w in t for w in ["worried", "scared", "urgent", "emergency", "stolen", "danger"]):
            scores["Fear / Anxiety"] += 0.4

        if comp > 0.3:
            scores["Joy / Gratitude"] += 0.3
        elif comp < -0.3:
            scores["Anger / Frustration"] += 0.25
            scores["Disappointment / Sadness"] += 0.15

        # Normalize to sum to 1.0
        total = sum(scores.values())
        return {k: round(v / total, 4) for k, v in scores.items()}

    def explain_prediction(self, text: str) -> List[Dict[str, Any]]:
        """Estimates token sensitivity by measuring leave-one-token-out probability changes."""
        tokens = text.split()
        if not tokens:
            return []

        base_res = self.predict(text)
        target_emotion = base_res["primary_emotion"]
        base_prob = base_res["probabilities"].get(target_emotion, 0.5)

        attributions = []
        for i in range(len(tokens)):
            # Leave-one-out perturbation
            perturbed_tokens = [tokens[j] for j in range(len(tokens)) if j != i]
            perturbed_text = " ".join(perturbed_tokens)
            if not perturbed_text.strip():
                continue
            perturbed_res = self.predict(perturbed_text)
            perturbed_prob = perturbed_res["probabilities"].get(target_emotion, 0.5)

            # Contribution = difference in probability
            delta = base_prob - perturbed_prob
            attributions.append({
                "token": tokens[i],
                "attribution": round(float(delta), 4),
                "direction": "Positive" if delta > 0 else "Negative"
            })

        # Sort by absolute impact
        attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        return attributions

    def check_data_drift(self, recent_texts: List[str]) -> Dict[str, Any]:
        """Calculates histogram PSI against the fixed TWCS reference sample."""
        if not recent_texts:
            return {"status": "INSUFFICIENT_DATA", "drift_detected": False}

        lengths = [len(t) for t in recent_texts]
        word_counts = [len(t.split()) for t in recent_texts]
        compounds = [self.get_sentiment(t)["compound"] for t in recent_texts]

        curr_mean_len = float(np.mean(lengths))
        curr_mean_words = float(np.mean(word_counts))
        curr_mean_comp = float(np.mean(compounds))

        baseline = self.drift_baseline
        base_len = baseline['mean_length']
        base_comp = baseline['mean_vader_compound']
        length_psi = self._population_stability_index(lengths, baseline['length'])
        sentiment_psi = self._population_stability_index(compounds, baseline['sentiment'])
        psi_score = max(length_psi, sentiment_psi)

        if psi_score > 0.25:
            drift_status = "CRITICAL DRIFT DETECTED"
            alert_color = "#DC2626"
            recommendation = "Significant distribution shift detected. Model retraining recommended via DVC pipeline."
        elif psi_score > 0.10:
            drift_status = "MODERATE DRIFT (WARNING)"
            alert_color = "#F59E0B"
            recommendation = "Input distributions show minor divergence. Monitor incoming customer support queues."
        else:
            drift_status = "STABLE (NO DRIFT)"
            alert_color = "#10B981"
            recommendation = "No significant shift in text length or sentiment from the TWCS reference sample."

        return {
            "status": drift_status,
            "drift_detected": psi_score > 0.10,
            "psi_score": round(psi_score, 4),
            "feature_psi": {"length": round(length_psi, 4), "sentiment": round(sentiment_psi, 4)},
            "baseline_source": baseline['source'],
            "alert_color": alert_color,
            "sample_size": len(recent_texts),
            "current_metrics": {
                "mean_char_length": round(curr_mean_len, 1),
                "mean_word_count": round(curr_mean_words, 1),
                "mean_sentiment_compound": round(curr_mean_comp, 3)
            },
            "baseline_metrics": {
                "mean_char_length": round(base_len, 1),
                "mean_word_count": round(baseline["mean_words"], 1),
                "mean_sentiment_compound": round(base_comp, 3)
            },
            "recommendation": recommendation
        }

    @staticmethod
    def _population_stability_index(values, reference):
        # Fixed bins and proportion-based smoothing make PSI invariant to
        # repetition of an otherwise identical sample, including empty bins.
        edges = [-np.inf, *reference['cut_points'], np.inf]
        actual = np.histogram(values, bins=edges)[0].astype(float)
        actual /= actual.sum()
        expected = np.asarray(reference['proportions'], dtype=float)
        actual = np.maximum(actual, 1e-6)
        expected = np.maximum(expected, 1e-6)
        actual /= actual.sum()
        expected /= expected.sum()
        return float(np.sum((actual - expected) * np.log(actual / expected)))


# Global engine instance
_ENGINE: Optional[EmotionEngine] = None

def get_engine() -> EmotionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = EmotionEngine()
    return _ENGINE
