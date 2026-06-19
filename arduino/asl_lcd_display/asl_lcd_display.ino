/*
  ASL Sentence Display - Arduino Nano + 16x2 I2C LCD
*/

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

#define SENTENCE_MAX 100
#define LINE_WIDTH 16
#define MAX_LINES 10

char sentence[SENTENCE_MAX + 1] = "";
int sentenceLen = 0;

char lines[MAX_LINES][LINE_WIDTH + 1];
int numLines = 0;

void setup() {
  Serial.begin(9600);
  lcd.init();
  lcd.backlight();
  rebuildLines();
  showLines();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '\f') {
      sentenceLen = 0;
    } else if (sentenceLen < SENTENCE_MAX) {
      sentence[sentenceLen++] = c;
    }
    sentence[sentenceLen] = '\0';

    rebuildLines();
    showLines();
  }
}

// Word-wraps `sentence` into `lines[]`, breaking only at spaces.
// A "word" longer than LINE_WIDTH gets hard-broken across lines.
void rebuildLines() {
  numLines = 1;
  lines[0][0] = '\0';
  int lineLen = 0;

  int i = 0;
  while (i < sentenceLen) {
    // skip any spaces (they only matter as separators between words)
    while (i < sentenceLen && sentence[i] == ' ') i++;
    if (i >= sentenceLen) break;

    int wordStart = i;
    while (i < sentenceLen && sentence[i] != ' ') i++;
    int wordLen = i - wordStart;

    while (wordLen > 0) {
      int sep = (lineLen > 0) ? 1 : 0;
      int avail = LINE_WIDTH - lineLen - sep;

      if (wordLen <= avail) {
        // Whole (remaining) word fits on the current line
        if (sep) lines[numLines - 1][lineLen++] = ' ';
        memcpy(&lines[numLines - 1][lineLen], &sentence[wordStart], wordLen);
        lineLen += wordLen;
        lines[numLines - 1][lineLen] = '\0';
        wordStart += wordLen;
        wordLen = 0;

      } else if (wordLen <= LINE_WIDTH && lineLen > 0) {
        // Doesn't fit here, but would fit on a fresh line -> move it down
        if (numLines < MAX_LINES) numLines++;
        lines[numLines - 1][0] = '\0';
        lineLen = 0;

      } else {
        // Word itself is too long for one line: hard-break it
        int take = avail > 0 ? avail : 0;
        if (take > 0) {
          if (sep) lines[numLines - 1][lineLen++] = ' ';
          memcpy(&lines[numLines - 1][lineLen], &sentence[wordStart], take);
          lineLen += take;
          lines[numLines - 1][lineLen] = '\0';
          wordStart += take;
          wordLen -= take;
        }
        if (wordLen > 0) {
          if (numLines < MAX_LINES) numLines++;
          lines[numLines - 1][0] = '\0';
          lineLen = 0;
        }
      }
    }
  }
}

// Shows the last two wrapped lines (so the display "scrolls" as the
// sentence grows). If there's only one line so far, it goes on the top row.
void showLines() {
  lcd.clear();

  if (numLines <= 1) {
    lcd.setCursor(0, 0);
    lcd.print(lines[0]);
  } else {
    lcd.setCursor(0, 0);
    lcd.print(lines[numLines - 2]);
    lcd.setCursor(0, 1);
    lcd.print(lines[numLines - 1]);
  }
}
