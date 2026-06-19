"""
Collect landmark samples for a custom gesture (e.g. your "SPACE" sign) using
your webcam, with the same normalization used for the main dataset. The
resulting CSV gets combined with the ASL letters CSV before training.

Controls:
    SPACE bar - toggle recording on/off
    q         - quit and save what's been collected

Usage:
    python collect_custom_sign.py --label SPACE \
        --output ../data/landmarks/custom_space.csv --target 300

Tips:
  - Hold a steady pose, press SPACE to start recording, then move your hand
    slightly (rotate/tilt a bit, move closer/farther) while holding the
    handshape so the model learns some natural variation.
  - Press SPACE again to pause, reposition, and resume if you want.
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp

from landmark_utils import normalize_landmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, help='Label for this gesture, e.g. SPACE')
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', type=int, default=300, help='Number of samples to collect')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                            min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(args.camera)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    recording = False
    samples = []

    print(f"Collecting samples for label '{args.label}'.")
    print("Hold your custom gesture, then press SPACE to start/stop recording.")
    print("Press 'q' when you've collected enough samples (or it'll stop automatically at --target).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # NOTE: deliberately not mirror-flipped - must match the dataset's
        # unflipped orientation, or the trained gesture won't line up with
        # the A-Z letters during recognition.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            if recording:
                coords = [(p.x, p.y, p.z) for p in hand_lm.landmark]
                features = normalize_landmarks(coords)
                samples.append(features)

        status = 'RECORDING' if recording else 'paused'
        cv2.putText(frame, f'{args.label}: {len(samples)}/{args.target} ({status})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Collect Custom Sign', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = not recording
        elif key == ord('q') or len(samples) >= args.target:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    write_header = not os.path.exists(args.output)
    with open(args.output, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
            writer.writerow(header)
        for features in samples:
            writer.writerow([args.label] + features.tolist())

    print(f'\nSaved {len(samples)} samples to {args.output}')


if __name__ == '__main__':
    main()
