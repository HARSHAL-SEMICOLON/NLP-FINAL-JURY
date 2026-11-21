/*
  Smart Academic Help – Arduino UNO R4 WiFi status display (board only, no wiring)
  ---------------------------------------------------------------------------------
  Uses only what is built into the UNO R4 WiFi:
    * 12x8 LED matrix  -> status icons + scrolling messages
    * on-board LED (L, pin 13) -> blinks while the assistant is processing
    * Wi-Fi module     -> tiny HTTP server the Python app talks to

  HTTP API (called by sah/arduino.py):
    GET /status?state=ready|processing|success|error&msg=Text  -> icon, then scrolls msg
                                                                  (msg is not scrolled while processing)
    GET /ping                                                  -> {"ok":true,"ip":"...","state":"...","uptime":s}
    GET /button                                                -> {"presses":0}  (no button in this build)

  Setup:
    1. Boards Manager: install "Arduino UNO R4 Boards".
    2. Library Manager: install "ArduinoGraphics".
    3. Copy arduino_secrets.h.example to arduino_secrets.h, put your 2.4 GHz Wi-Fi name/password in it, upload.
    4. Serial Monitor (115200): copy the printed IP into the app's .env as ARDUINO_IP=...
*/

#include "ArduinoGraphics.h"   // must be included BEFORE Arduino_LED_Matrix.h
#include "Arduino_LED_Matrix.h"
#include <WiFiS3.h>

// ---------- edit these ----------
#include "arduino_secrets.h"           // Wi-Fi name/password live here (not on GitHub)
const char WIFI_SSID[] = SECRET_SSID;
const char WIFI_PASS[] = SECRET_PASS;
// ---------------------------------

ArduinoLEDMatrix matrix;
WiFiServer server(80);

String state = "ready";
String message = "READY";
unsigned long lastBlinkMs = 0;
bool blinkOn = false;

// 8 x 12 icons for the LED matrix (1 = LED on)
uint8_t ICON_READY[8][12] = {      // smiley
  {0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},
  {0,0,1,0,0,0,0,0,0,1,0,0},{0,0,0,1,0,0,0,0,1,0,0,0},{0,0,0,0,1,1,1,1,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}};
uint8_t ICON_BUSY[8][12] = {       // hourglass
  {0,0,0,1,1,1,1,1,1,0,0,0},{0,0,0,0,1,0,0,1,0,0,0,0},{0,0,0,0,0,1,1,0,0,0,0,0},{0,0,0,0,0,1,1,0,0,0,0,0},
  {0,0,0,0,0,1,1,0,0,0,0,0},{0,0,0,0,1,1,1,1,0,0,0,0},{0,0,0,1,1,1,1,1,1,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}};
uint8_t ICON_BUSY2[8][12] = {      // hourglass, sand fallen (animation frame 2)
  {0,0,0,1,1,1,1,1,1,0,0,0},{0,0,0,0,1,1,1,1,0,0,0,0},{0,0,0,0,0,1,1,0,0,0,0,0},{0,0,0,0,0,1,1,0,0,0,0,0},
  {0,0,0,0,0,1,1,0,0,0,0,0},{0,0,0,0,1,0,0,1,0,0,0,0},{0,0,0,1,1,1,1,1,1,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}};
uint8_t ICON_OK[8][12] = {         // tick
  {0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,1,0,0},{0,0,0,0,0,0,0,0,1,0,0,0},{0,0,0,0,0,0,0,1,0,0,0,0},
  {0,0,1,0,0,0,1,0,0,0,0,0},{0,0,0,1,0,1,0,0,0,0,0,0},{0,0,0,0,1,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}};
uint8_t ICON_ERR[8][12] = {        // cross
  {0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0},{0,0,0,0,1,0,0,1,0,0,0,0},{0,0,0,0,0,1,1,0,0,0,0,0},
  {0,0,0,0,0,1,1,0,0,0,0,0},{0,0,0,0,1,0,0,1,0,0,0,0},{0,0,0,1,0,0,0,0,1,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}};
uint8_t ICON_WIFI[8][12] = {       // Wi-Fi arcs (connecting)
  {0,0,0,1,1,1,1,1,1,0,0,0},{0,0,1,0,0,0,0,0,0,1,0,0},{0,1,0,0,1,1,1,1,0,0,1,0},{0,0,0,1,0,0,0,0,1,0,0,0},
  {0,0,0,0,0,1,1,0,0,0,0,0},{0,0,0,0,0,1,1,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}};

void showState() {
  digitalWrite(LED_BUILTIN, LOW);
  if (state == "ready")           matrix.renderBitmap(ICON_READY, 8, 12);
  else if (state == "processing") matrix.renderBitmap(ICON_BUSY, 8, 12);
  else if (state == "success")    matrix.renderBitmap(ICON_OK, 8, 12);
  else                            matrix.renderBitmap(ICON_ERR, 8, 12);
}

void scrollText(const String &text) {
  matrix.beginDraw();
  matrix.stroke(0xFFFFFFFF);
  matrix.textScrollSpeed(60);
  matrix.textFont(Font_5x7);
  matrix.beginText(0, 1, 0xFFFFFF);
  matrix.println("  " + text + "  ");
  matrix.endText(SCROLL_LEFT);       // blocks while the text scrolls (~2-4 s)
  matrix.endDraw();
}

// Read a query-string parameter and URL-decode it (+ and %XX).
String param(const String &query, const String &key) {
  int start = query.indexOf(key + "=");
  if (start < 0) return "";
  start += key.length() + 1;
  int end = query.indexOf('&', start);
  String raw = query.substring(start, end < 0 ? query.length() : end);
  String out = "";
  for (unsigned int i = 0; i < raw.length(); i++) {
    char c = raw[i];
    if (c == '+') out += ' ';
    else if (c == '%' && i + 2 < raw.length()) {
      out += (char) strtol(raw.substring(i + 1, i + 3).c_str(), NULL, 16);
      i += 2;
    } else out += c;
  }
  return out;
}

void sendJson(WiFiClient &client, const String &body, int code = 200) {
  client.print("HTTP/1.1 ");
  client.print(code);
  client.println(code == 200 ? " OK" : " Bad Request");
  client.println("Content-Type: application/json");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Connection: close");
  client.print("Content-Length: ");
  client.println(body.length());
  client.println();
  client.print(body);
}

void handleClient(WiFiClient &client) {
  String requestLine = "";
  unsigned long t0 = millis();
  // read the first line: "GET /status?state=processing HTTP/1.1"
  while (client.connected() && millis() - t0 < 1000) {
    if (client.available()) {
      char c = client.read();
      if (c == '\n') break;
      if (c != '\r') requestLine += c;
    }
  }
  while (client.available()) client.read();  // discard headers

  int s1 = requestLine.indexOf(' ');
  int s2 = requestLine.indexOf(' ', s1 + 1);
  String target = requestLine.substring(s1 + 1, s2);
  String path = target, query = "";
  int q = target.indexOf('?');
  if (q >= 0) { path = target.substring(0, q); query = target.substring(q + 1); }

  bool scrollAfter = false;
  if (path == "/status") {
    String st = param(query, "state");
    if (st == "ready" || st == "processing" || st == "success" || st == "error") {
      state = st;
      String m = param(query, "msg");
      message = m;
      message.toUpperCase();
      showState();
      sendJson(client, "{\"ok\":true,\"state\":\"" + state + "\"}");
      // Scrolling blocks the loop, so never scroll while busy: the "done" call
      // must be answered immediately. Busy state shows the animated hourglass.
      scrollAfter = (state != "processing" && m.length() > 0);
    } else {
      sendJson(client, "{\"ok\":false,\"error\":\"bad state\"}", 400);
    }
  } else if (path == "/ping") {
    sendJson(client, "{\"ok\":true,\"ip\":\"" + WiFi.localIP().toString() + "\",\"state\":\"" + state +
                     "\",\"uptime\":" + String(millis() / 1000) + "}");
  } else if (path == "/button") {
    sendJson(client, "{\"presses\":0}");   // no physical button in this build
  } else {
    sendJson(client, "{\"ok\":false,\"error\":\"unknown path\"}", 400);
  }
  delay(2);
  client.stop();
  if (scrollAfter) {                     // reply first so Python is not kept waiting
    scrollText(message);
    showState();                         // back to the state icon
  }
}

void connectWiFi() {
  matrix.renderBitmap(ICON_WIFI, 8, 12);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print("Connecting to ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) delay(500);
  }
  // DHCP can report 0.0.0.0 for a moment after connecting
  while (WiFi.localIP() == IPAddress(0, 0, 0, 0)) delay(200);
  Serial.print("Connected. Put this in .env ->  ARDUINO_IP=");
  Serial.println(WiFi.localIP());
  server.begin();
  state = "ready";
  scrollText(WiFi.localIP().toString());
  showState();
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  matrix.begin();
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("WiFi module not found");
    matrix.renderBitmap(ICON_ERR, 8, 12);
    while (true) {}
  }
  connectWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();

  WiFiClient client = server.available();
  if (client) handleClient(client);

  // While processing: animate the hourglass and blink the on-board LED.
  // renderBitmap is a macro (&bitmap[0][0]), so it needs a plain array name – no ?: inside it.
  if (state == "processing" && millis() - lastBlinkMs > 400) {
    lastBlinkMs = millis();
    blinkOn = !blinkOn;
    if (blinkOn) matrix.renderBitmap(ICON_BUSY2, 8, 12);
    else         matrix.renderBitmap(ICON_BUSY, 8, 12);
    digitalWrite(LED_BUILTIN, blinkOn ? HIGH : LOW);
  }
}