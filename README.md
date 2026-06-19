# ASL Alphabet Recognizer -> Arduino LCD

Real-time ASL fingerspelling recognition from a webcam, with a custom
"space" gesture, that sends each recognized letter to an Arduino Nano
driving a 16x2 I2C LCD so it displays your spelled-out sentence with
proper word wrapping.

## How it works

```
Webcam -> MediaPipe (21 hand landmarks) -> normalize -> Random Forest
       -> stability/debounce logic -> serial -> Arduino Nano -> I2C LCD
```

- **MediaPipe Hands** extracts 21 (x, y, z) landmark points per hand.
- Landmarks are **normalized** (translated so the wrist is the origin,
  scaled by hand size) so the model works regardless of where your hand
  is in frame or how close it is to the camera.
- A **Random Forest** classifier (trained on landmark coordinates, not
  raw images) predicts one of 27 classes: A-Z plus your custom SPACE
  gesture.
- A small **state machine** only "commits" a letter once it's been held
  steadily for ~10 frames, and won't re-fire while you keep holding the
  same pose.
- Confirmed characters are sent one at a time over **USB serial** to the
  Nano, which word-wraps the growing sentence across the 16x2 display.

## 1. Hardware setup

Wire the I2C LCD backpack to the Nano:

| LCD pin | Nano pin |
|---------|----------|
| VCC     | 5V       |
| GND     | GND      |
| SDA     | A4       |
| SCL     | A5       |

No extra wiring is needed for serial - the same USB cable you use to
program the Nano is what `realtime_recognition.py` talks over.

## 2. Python setup

```bash
cd asl-lcd-project
pip install -r requirements.txt
```

## 3. Collect training data from your own webcam

This project trains entirely on self-collected data: every letter A-Z
plus a custom SPACE gesture, recorded from your own webcam using
`collect_custom_sign.py`. This keeps the dataset consistent with the
exact camera, lighting, and hand you'll be using at inference time.

```bash
cd scripts
python collect_custom_sign.py --label A --output ../data/landmarks/my_samples.csv --target 150
python collect_custom_sign.py --label B --output ../data/landmarks/my_samples.csv --target 150
# ...repeat for every letter A-Z...
python collect_custom_sign.py --label SPACE --output ../data/landmarks/my_samples.csv --target 150
```

All labels are appended into the same CSV (`my_samples.csv`), so each
command above can be run independently, with a short break between
letters to reposition your hand. For each one: hold the handshape,
press SPACE to start recording, move/rotate your hand slightly while
recording so the model sees some natural variation, and press SPACE
again to pause or `q` to stop early.

For your SPACE gesture, pick a handshape that isn't used anywhere in
the ASL alphabet - the "ILY" sign (thumb, index finger, and pinky
extended; middle and ring folded down) works well since it's
distinctive and easy to hold.

## 4. Train the classifier

```bash
python train_model.py \
    --inputs ../data/landmarks/my_samples.csv \
    --output ../models/asl_classifier.pkl
```

This prints a classification report so you can see per-letter accuracy.
If a particular letter performs poorly, it's often because of confusable
handshapes (e.g. M/N/S/T, or U/V/K) - try collecting more samples for
just that letter and re-running training (you can pass multiple `--inputs`
files if you split your recordings across several CSVs).

> **Note:** `extract_landmarks.py` also exists in this project if you
> ever want to bootstrap with the public Kaggle ASL Alphabet dataset
> (https://www.kaggle.com/datasets/grassknoted/asl-alphabet) alongside
> your own samples, but it isn't required - the model here is trained
> purely on self-collected webcam data.

## 7. Flash the Arduino

1. Open `arduino/asl_lcd_display/asl_lcd_display.ino` in the Arduino IDE.
2. Install the **"LiquidCrystal I2C" by Frank de Brabander** library via
   Library Manager.
3. If the screen stays blank after uploading, your backpack might use
   address `0x3F` instead of `0x27` - change the line
   `LiquidCrystal_I2C lcd(0x27, 16, 2);` accordingly (run an I2C scanner
   sketch if you're not sure).
4. Upload, then **close the Arduino IDE's Serial Monitor** - only one
   program can use the serial port at a time, and Python needs it.

## 8. Run real-time recognition

Find your Arduino's serial port:
- Windows: Device Manager -> Ports (COM & LPT), e.g. `COM3`
- macOS: `/dev/cu.usbserial-XXXX` or similar
- Linux: `/dev/ttyUSB0` or `/dev/ttyACM0`

```bash
python realtime_recognition.py --model ../models/asl_classifier.pkl --port COM3
```

Or test without the Arduino first:
```bash
python realtime_recognition.py --model ../models/asl_classifier.pkl
```

Controls: `q` to quit, `c` to clear the sentence (also clears the LCD).

## Tuning & known limitations

- **J and Z** are the only ASL alphabet letters that involve *motion*
  rather than a static handshape. This system classifies a single frame
  at a time, so it'll recognize a static approximation of J/Z but won't
  capture the motion. A future improvement could track a short window of
  landmark positions over time for these two letters specifically.
- **Double letters** (e.g. "LL") require briefly relaxing your hand
  between signs - the state machine waits for the pose to "release"
  before it'll commit the same letter again.
- `WINDOW_SIZE` (frames required to confirm a letter) and
  `CONF_THRESHOLD` (minimum model confidence) are constants near the top
  of `realtime_recognition.py` - increase `WINDOW_SIZE` if letters fire
  too eagerly, decrease it if the system feels sluggish.

## Possible extensions

- A "backspace"/"delete" gesture to correct misread letters.
- A "new sentence" gesture mapped to the clear signal (`\f`) instead of
  needing the `c` key.
- Increased vocabulary input
