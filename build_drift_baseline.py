"""Build aggregate drift reference data from a TWCS CSV without retaining messages.

Usage: python build_drift_baseline.py path/to/twcs_sample_5k.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def histogram(values, cut_points):
    counts = np.histogram(values, bins=[-np.inf, *cut_points, np.inf])[0]
    return {'cut_points': cut_points, 'proportions': (counts / counts.sum()).tolist()}


def build(csv_path):
    texts = pd.read_csv(csv_path)['clean_text'].fillna('').astype(str).tolist()
    if not texts:
        raise ValueError('Reference CSV must contain messages')
    sia = SentimentIntensityAnalyzer()
    lengths = [len(t) for t in texts]
    compounds = [sia.polarity_scores(t)['compound'] for t in texts]
    return {
        'source': f'TWCS reference sample: {Path(csv_path).name}',
        'sample_size': len(texts),
        'mean_length': float(np.mean(lengths)),
        'mean_words': float(np.mean([len(t.split()) for t in texts])),
        'mean_vader_compound': float(np.mean(compounds)),
        'length': histogram(lengths, [25, 50, 75, 100, 125, 150, 200, 280]),
        'sentiment': histogram(compounds, [-0.75, -0.5, -0.25, -0.05, 0.05, 0.25, 0.5, 0.75]),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path')
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / 'models' / 'drift_baseline.json'
    output.write_text(json.dumps(build(args.csv_path), indent=2) + '\n', encoding='utf-8')
