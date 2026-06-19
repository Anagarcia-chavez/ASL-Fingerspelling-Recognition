"""
Extract MediaPipe hand landmarks from the ASL Alphabet image dataset
(https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and save them
as a CSV of normalized landmark features + letter labels.

Expected folder structure after downloading & extracting the dataset:

    data/raw/asl_alphabet_train/asl_alphabet_train/A/*.jpg
    data/raw/asl_alphabet_train/asl_alphabet_train/B/*.jpg
    ...
    data/raw/asl_alphabet_train/asl_alphabet_train/Z/*.jpg

Usage:
    python extract_landmarks.py \
        --data_dir ../data/raw/asl_alphabet_train/asl_alphabet_train \
        --output ../data/landmarks/asl_landmarks.csv \
        --max_per_class 300

You don't need all ~3000 images per letter - a few hundred per class is
plenty for a landmark-based Random Forest and keeps processing fast.
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp

from landmark_utils import normalize_landmarks

LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                         help='Path to folder containing A/, B/, ..., Z/ subfolders of images')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--max_per_class', type=int, default=300,
                         help='Max images to process per letter')
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                            min_detection_confidence=0.5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rows_written = 0
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
        writer.writerow(header)

        for letter in LETTERS:
            folder = os.path.join(args.data_dir, letter)
            if not os.path.isdir(folder):
                print(f'Skipping {letter}: folder not found at {folder}')
                continue

            files = sorted(os.listdir(folder))[:args.max_per_class]
            kept = 0
            for fname in files:
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if not results.multi_hand_landmarks:
                    continue

                lm = results.multi_hand_landmarks[0].landmark
                coords = [(p.x, p.y, p.z) for p in lm]
                features = normalize_landmarks(coords)

                writer.writerow([letter] + features.tolist())
                kept += 1
                rows_written += 1

            print(f'{letter}: kept {kept}/{len(files)} images (hand detected)')

    hands.close()
    print(f'\nDone. Wrote {rows_written} rows to {args.output}')


if __name__ == '__main__':
    main()
