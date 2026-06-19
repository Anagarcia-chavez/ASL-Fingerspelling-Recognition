"""
Real-time ASL alphabet recognition with Arduino LCD output.

What this does each frame:
  1. Grabs a webcam frame and runs MediaPipe Hands on it.
  2. Normalizes the 21 landmarks (same way as during training) and runs
     them through the trained classifier.
  3. Feeds the prediction into a small "stability" state machine so a
     letter only gets committed once you've HELD a pose steadily, and
     won't repeat-fire while you keep holding it.
  4. Sends each newly committed letter over serial to an Arduino Nano,
     which displays the growing sentence on a 16x2 I2C LCD.

Your custom gesture (trained with label "SPACE") gets sent as a literal
space character, which the Arduino uses as a word boundary for wrapping.

Usage:
    python realtime_recognition.py --model ../models/asl_classifier.pkl --port COM3

    (Omit --port to test the recognizer without an Arduino connected.)

Controls:
    q - quit
    c - clear the current sentence (also tells the Arduino to clear the LCD)

Tuning:
    WINDOW_SIZE     - how many consecutive frames must agree before a
                       letter is committed. Higher = more deliberate /
                       fewer accidental letters, but slower to respond.
    CONF_THRESHOLD  - minimum model confidence to accept a prediction at all.

Notes on double letters:
    To sign the same letter twice in a row (e.g. "LL"), briefly relax/move
    your hand out of the pose between the two signs. The state machine
    requires the pose to "release" before the same letter can commit again.
"""

import argparse
import collections
import time

import cv2
import joblib
import mediapipe as mp
import serial

from landmark_utils import normalize_landmarks

WINDOW_SIZE = 10
CONF_THRESHOLD = 0.6

# Control character sent to the Arduino to mean "clear the sentence/display"
CLEAR_SIGNAL = '\f'  # ASCII form feed (0x0C)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained .pkl model')
    parser.add_argument('--port', default=None,
                         help='Serial port for Arduino, e.g. COM3 or /dev/ttyUSB0. '
                              'Omit to run without sending to Arduino.')
    parser.add_argument('--baud', type=int, default=9600)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    clf = joblib.load(args.model)

    ser = None
    if args.port:
        ser = serial.Serial(args.port, args.baud)
        time.sleep(2)  # let the Nano finish resetting after the serial port opens
        print(f'Connected to Arduino on {args.port}')

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                            min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(args.camera)

    history = collections.deque(maxlen=WINDOW_SIZE)
    last_committed = None
    released = True
    sentence = ''

    print("Running. Press 'q' to quit, 'c' to clear the sentence.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # NOTE: we deliberately do NOT mirror-flip the frame here. The
        # training dataset images are unflipped, so landmarks must be
        # computed in that same (unflipped) orientation for the model to
        # recognize letters correctly. The video will look like a
        # non-mirror camera (your right hand appears on the right side of
        # the screen) - this is expected.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_label = None
        current_conf = 0.0

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            coords = [(p.x, p.y, p.z) for p in hand_lm.landmark]
            features = normalize_landmarks(coords).reshape(1, -1)

            probs = clf.predict_proba(features)[0]
            best_idx = probs.argmax()
            current_conf = probs[best_idx]
            if current_conf >= CONF_THRESHOLD:
                current_label = clf.classes_[best_idx]

        history.append(current_label)

        # A label is "stable" if the entire history window agrees on it
        stable_label = None
        if (len(history) == WINDOW_SIZE
                and history.count(history[0]) == WINDOW_SIZE
                and history[0] is not None):
            stable_label = history[0]

        if stable_label is None:
            if current_label != last_committed:
                released = True
        else:
            if stable_label != last_committed or released:
                char_to_send = ' ' if stable_label == 'SPACE' else stable_label
                sentence += char_to_send
                last_committed = stable_label
                released = False

                if ser:
                    ser.write(char_to_send.encode())

                print(f'Committed: {char_to_send!r} -> "{sentence}"')

        display_label = current_label if current_label else '-'
        cv2.putText(frame, f'Detected: {display_label} ({current_conf:.2f})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Sentence: {sentence[-30:]}',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('ASL Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = ''
            last_committed = None
            released = True
            if ser:
                ser.write(CLEAR_SIGNAL.encode())

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    if ser:
        ser.close()


if __name__ == '__main__':
    main()
