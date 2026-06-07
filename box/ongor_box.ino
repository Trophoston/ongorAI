// ongor_box.ino — Ong-Or Box (ฝั่ง MCU / ภาษา C)
// หน้าที่: เมนู (knob), คาลิเบรตระยะ (distance), แสดงผล (LCD/Pixels/Buzzer)
//          และ "รับสถานะจากฝั่ง Python ผ่าน Bridge" มาแสดงให้ผู้เล่น
//
// โปรโตคอลกับ Python:
//   - sendMode(m)  : บอก Python ว่าผู้เล่นเลือกโหมดอะไร (1=Play, 2=Test, 3=CamCheck, 0=เมนู)
//   - poll()       : ขอข้อความแสดงผลจาก Python -> "l1|l2|pix|buz"
//                    l1,l2 = 2 บรรทัดบนจอ, pix = จำนวนไฟ 0-8, buz = เสียง 1-4
//
// *** logic เกมทั้งหมดอยู่ฝั่ง Python (box/main.py) — ฝั่งนี้แค่แสดงผล ***

#include <Arduino_Modulino.h>
#include <IskakINO_LiquidCrystal_I2C.h>

// ====== อุปกรณ์ ======
ModulinoKnob      knob;
ModulinoBuzzer    buzzer;
ModulinoPixels    pixels;
ModulinoDistance  distance;
LiquidCrystal_I2C lcd(16, 2);

// ====== CONFIG ======
const uint8_t LED_COUNT  = 8;
const uint8_t LED_BRIGHT = 30;

float DIST_NEAR = 80.0;
float DIST_FAR  = 150.0;
const unsigned long HOLD_MS = 1500;
const unsigned long POLL_MS = 150;

#define COL_RED   0
#define COL_GREEN 1
#define COL_BLUE  2

// ====== เมนู ======
const char* MENU[] = { "1.Play Game", "2.Test AI", "3.Cam Check" };
const int   MENU_N = 3;

// ====== สถานะ ======
enum AppState { ST_MENU, ST_CALIB, ST_RUN };
AppState state = ST_MENU;

int  menuIndex = 0;
int  lastKnob = 0;
bool lastPressed = false;
int  selectedMode = 0;                 // 1=play, 2=test, 3=camcheck

unsigned long lastPoll = 0;
unsigned long calibInRangeSince = 0;
String runPrevL1 = "", runPrevL2 = "";

// ====== Pixels ======
void setBar(uint8_t n, int colorId) {
  for (uint8_t i = 0; i < LED_COUNT; ++i) {
    if (i < n) {
      switch (colorId) {
        case COL_RED:  pixels.set(i, RED,   LED_BRIGHT); break;
        case COL_BLUE: pixels.set(i, BLUE,  LED_BRIGHT); break;
        default:       pixels.set(i, GREEN, LED_BRIGHT); break;
      }
    } else pixels.clear(i);
  }
  pixels.show();
}
void clearPixels() { setBar(0, COL_GREEN); }

// ====== เสียง (1=tick, 2=ถูก, 3=ชนะ, 4=ผิด/แพ้) ======
void cue(int c) {
  switch (c) {
    case 1: buzzer.tone(700, 40);  break;
    case 2: buzzer.tone(1000, 80); break;
    case 3: buzzer.tone(880,120); delay(130); buzzer.tone(1175,120); delay(130); buzzer.tone(1568,220); break;
    case 4: buzzer.tone(220, 300); break;
    default: break;
  }
}

// ====== จอ ======
void lcdShow(const String& l1, const String& l2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(l1);
  lcd.setCursor(0, 1); lcd.print(l2);
}
void drawMenu() {
  lcdShow(">" + String(MENU[menuIndex]),
          " " + String(MENU[(menuIndex + 1) % MENU_N]));
  setBar(menuIndex + 1, COL_BLUE);
}

// ====== Distance ======
float readDistance() {
  static float last = -1;
  if (distance.available()) {
    last = distance.get();
    Serial.print("dist="); Serial.println(last);
  }
  return last;
}

// ====== Bridge ======
void sendMode(int mode) {
  String r;
  Bridge.call("set_mode", mode).result(r);
}

void enterCalib();
void enterRun();

// ====== Setup ======
void setup() {
  Serial.begin(9600);
  Bridge.begin();
  Modulino.begin();
  knob.begin();
  buzzer.begin();
  pixels.begin();
  distance.begin();
  lcd.begin();
  lcd.backlight();

  lastKnob = knob.get();
  lastPressed = knob.isPressed();

  lcdShow("  Ong-Or Box", "  starting...");
  buzzer.tone(1000, 150);
  delay(1000);

  state = ST_MENU;
  drawMenu();
}

// ====== MENU ======
void loopMenu() {
  int  val = knob.get();
  bool pressed = knob.isPressed();

  if (val != lastKnob) {
    if (val > lastKnob) menuIndex = (menuIndex + 1) % MENU_N;
    else                menuIndex = (menuIndex - 1 + MENU_N) % MENU_N;
    cue(1);
    drawMenu();
    lastKnob = val;
  }
  if (pressed && !lastPressed) {
    selectedMode = menuIndex + 1;       // 1/2/3
    cue(2);
    if (selectedMode == 3) {            // Cam Check = ข้ามคาลิเบรต เข้าเลย
      sendMode(3);
      enterRun();
    } else {                            // Play / Test = คาลิเบรตระยะก่อน
      enterCalib();
    }
  }
  lastPressed = pressed;
}

// ====== CALIBRATE ======
void enterCalib() {
  state = ST_CALIB;
  calibInRangeSince = 0;
  lcdShow("Calibrate dist", "Stand in front");
  delay(400);
}
void loopCalib() {
  bool pressed = knob.isPressed();
  if (pressed && !lastPressed) {
    lastPressed = pressed;
    cue(4);
    state = ST_MENU;
    lastKnob = knob.get();
    drawMenu();
    return;
  }
  lastPressed = pressed;

  float d = readDistance();
  if (d < 0) {
    lcdShow("Dist sensor", "no reading...");
    setBar(1, COL_RED); calibInRangeSince = 0;
  } else if (d > DIST_FAR) {
    lcdShow("Too far", "Move CLOSER >>");
    setBar(2, COL_RED); calibInRangeSince = 0;
  } else if (d < DIST_NEAR) {
    lcdShow("Too close", "<< Move BACK");
    setBar(8, COL_RED); calibInRangeSince = 0;
  } else {
    if (calibInRangeSince == 0) calibInRangeSince = millis();
    unsigned long held = millis() - calibInRangeSince;
    int pct = (int)min((unsigned long)100, held * 100 / HOLD_MS);
    lcdShow("Good distance", "Hold " + String(pct) + "%");
    setBar((uint8_t)map(pct, 0, 100, 1, 8), COL_GREEN);
    if (held >= HOLD_MS) {
      cue(3);
      sendMode(selectedMode);
      enterRun();
    }
  }
  delay(80);
}

// ====== RUN ======
void enterRun() {
  state = ST_RUN;
  lastPoll = 0;
  runPrevL1 = ""; runPrevL2 = "";
  lastPressed = knob.isPressed();
  String sub = selectedMode == 1 ? "Play game..."
             : selectedMode == 2 ? "Test AI..."
             :                      "Cam check...";
  lcdShow("Starting mode", sub);
  delay(500);
}
void handlePoll(const String& s) {
  int p1 = s.indexOf('|');
  int p2 = s.indexOf('|', p1 + 1);
  int p3 = s.indexOf('|', p2 + 1);
  if (p1 < 0 || p2 < 0 || p3 < 0) return;

  String l1 = s.substring(0, p1);
  String l2 = s.substring(p1 + 1, p2);
  int pix   = s.substring(p2 + 1, p3).toInt();
  int buz   = s.substring(p3 + 1).toInt();

  if (l1 != runPrevL1 || l2 != runPrevL2) {
    lcdShow(l1, l2);
    runPrevL1 = l1; runPrevL2 = l2;
  }
  setBar((uint8_t)constrain(pix, 0, LED_COUNT), COL_GREEN);
  if (buz > 0) cue(buz);
}
void loopRun() {
  bool pressed = knob.isPressed();
  if (pressed && !lastPressed) {
    lastPressed = pressed;
    sendMode(0);
    cue(4);
    state = ST_MENU;
    clearPixels();
    lastKnob = knob.get();
    drawMenu();
    return;
  }
  lastPressed = pressed;

  if (millis() - lastPoll >= POLL_MS) {
    lastPoll = millis();
    String resp;
    if (Bridge.call("poll").result(resp)) handlePoll(resp);
  }
}

// ====== Loop ======
void loop() {
  switch (state) {
    case ST_MENU:  loopMenu();  break;
    case ST_CALIB: loopCalib(); break;
    case ST_RUN:   loopRun();   break;
  }
  delay(20);
}
