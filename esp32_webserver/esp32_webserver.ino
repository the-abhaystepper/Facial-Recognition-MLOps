#include "esp_camera.h"
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <ESPmDNS.h>
#include "model_data.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "labels.h"

const char* ssid = "abhayhotspot";
const char* password = "abhay123";


AsyncWebServer server(80);

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

const int kArenaSize = 60 * 1024; 
uint8_t tensor_arena[kArenaSize];

uint8_t* frameBuffer = NULL;
size_t frameLen = 0;
bool frameReceived = false;

const uint8_t bmpHeader[54] = {
  0x42, 0x4D,             
  0x36, 0x24, 0x00, 0x00, 
  0x00, 0x00, 0x00, 0x00, 
  0x36, 0x04, 0x00, 0x00, 

  // DIB Header
  0x28, 0x00, 0x00, 0x00, // Header Size
  0x60, 0x00, 0x00, 0x00, // Width (96)
  0xA0, 0xFF, 0xFF, 0xFF, // Height (-96) -> Top-down
  0x01, 0x00,             // Planes (1)
  0x08, 0x00,             // Bits per pixel (8)
  0x00, 0x00, 0x00, 0x00, // Compression (BI_RGB = 0)
  0x00, 0x24, 0x00, 0x00, // Image Size (96*96)
  0x00, 0x00, 0x00, 0x00, // X Pixels per meter
  0x00, 0x00, 0x00, 0x00, // Y Pixels per meter
  0x00, 0x01, 0x00, 0x00, // Colors used (256)
  0x00, 0x00, 0x00, 0x00  // Important colors
};

uint8_t grayscalePalette[1024];

// Inference & Score Globals
volatile bool isInferencing = false;
volatile bool frameReadyForInference = false;
String lastResult = "Searching...";
float lastConfidence = 0.0;
// We will store scores dynamically in the loop, but for global access (JSON), we need a buffer
// Assuming max 10 classes for this simple example
int lastScores[10] = {0}; 
String lastSupervisoryMessage = "System Active"; 

const int LED_PIN = 14; 
const int LOCKDOWN_PIN = 32; 

void setup() {
  Serial.begin(115200);
  while(!Serial); 
  Serial.println("\n--- ESP32 TinyML Face Recognition ---");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 
  pinMode(LOCKDOWN_PIN, OUTPUT);
  digitalWrite(LOCKDOWN_PIN, LOW); 
  
  // Initialize Palette (Gray scale ramp)
  for(int i=0; i<256; i++){
    grayscalePalette[i*4+0] = i; // B
    grayscalePalette[i*4+1] = i; // G
    grayscalePalette[i*4+2] = i; // R
    grayscalePalette[i*4+3] = 0; // Padding
  }
  
  // ... TFLite Setup ...
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.print("Model Loaded. Input shape: ");
  Serial.print(input->dims->data[1]); Serial.print("x");
  Serial.print(input->dims->data[2]); Serial.println();

  Serial.println("Connecting to WiFi...");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  if (!MDNS.begin("esp32")) {
    Serial.println("Error setting up MDNS responder!");
  } else {
    Serial.println("mDNS responder started (http://esp32.local)");
    MDNS.addService("http", "tcp", 80);
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  Serial.println("Hostname: http://esp32.local");

  server.on("/upload", HTTP_POST, [](AsyncWebServerRequest *request){
    String responseText = lastResult + ":" + String(lastConfidence, 1);
    request->send(200, "text/plain", responseText);
  }, NULL, [](AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total){
    static int input_idx = 0;
    if(!index){
      input_idx = 0;
      if (frameBuffer == NULL || total > frameLen) {
        if (frameBuffer) free(frameBuffer);
        frameBuffer = (uint8_t*)malloc(total);
      }
      frameLen = total;
    }
    if (frameBuffer) memcpy(frameBuffer + index, data, len);
    if (frameBuffer) memcpy(frameBuffer + index, data, len);
    // Processing moved to loop() to avoid blocking the network callback
    if(index + len == total){
      frameReceived = true;
      if(!isInferencing) frameReadyForInference = true;
    }
  });

  server.on("/status", HTTP_GET, [](AsyncWebServerRequest *request){
    String json = "{";
    json += "\"label\":\"" + lastResult + "\",";
    json += "\"message\":\"" + lastSupervisoryMessage + "\",";
    json += "\"confidence\":" + String(lastConfidence, 1) + ",";
    json += "\"scores\":{";
    for(int i=0; i<kNumClasses; i++){
        json += "\"" + String(kCategoryLabels[i]) + "\":" + String(lastScores[i]);
        if(i < kNumClasses - 1) json += ",";
    }
    json += "}}";
    request->send(200, "application/json", json);
  });

  server.on("/set_message", HTTP_GET, [](AsyncWebServerRequest *request){
    if (request->hasParam("msg")) {
        lastSupervisoryMessage = request->getParam("msg")->value();
    }
    request->send(200, "text/plain", "OK");
  });

  server.on("/set_lockdown", HTTP_GET, [](AsyncWebServerRequest *request){
    if (request->hasParam("state")) {
        String state = request->getParam("state")->value();
        if (state == "1") {
            digitalWrite(LOCKDOWN_PIN, HIGH);
            Serial.println("LOCKDOWN LED ON");
        } else {
            digitalWrite(LOCKDOWN_PIN, LOW);
            Serial.println("LOCKDOWN LED OFF");
        }
    }
    request->send(200, "text/plain", "OK");
  });

  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    String html = "<html><head><title>ESP32 Face Recognition</title>";
    html += "<style>body{font-family:sans-serif; text-align:center; padding:20px;} img{width:400px; border:2px solid #555; background:#000; image-rendering:pixelated;}</style>";
    html += "</head><body>";
    html += "<h1>ESP32 Face Recognition</h1>";
    html += "<img src='/live' id='stream'><br>";
    html += "<h2>Result: <span id='label'>...</span></h2>";
    html += "<p>Confidence: <span id='conf'>0</span>%</p>";
    html += "<div style='margin:10px; padding:10px; background:#222; color:#0f0; border:1px solid #444; min-height:40px;' id='log'>System Active</div>";
    html += "<div id='score-list'></div>";
    html += "<script>";
    html += "const img = document.getElementById('stream');";
    html += "function refresh() { img.src = '/live?t=' + Date.now(); }";
    html += "img.onload = refresh;";
    html += "img.onerror = refresh;";
    html += "refresh();";
    html += "setInterval(async () => {";
    html += "  try { const r = await fetch('/status'); const d = await r.json();";
    html += "    document.getElementById('label').innerText = d.label;";
    html += "    document.getElementById('conf').innerText = d.confidence;";
    html += "    document.getElementById('log').innerText = d.message;";
    html += "    const score = d.scores[d.label];";
    html += "    document.getElementById('score-list').innerHTML = '<p><b>' + d.label + ' Score:</b> ' + score + '</p>';";
    html += "  } catch(e) {}";
    html += "}, 300);";
    html += "</script></body></html>";
    request->send(200, "text/html", html);
  });
  
  // ... /live endpoint (unchanged) ...
  server.on("/live", HTTP_GET, [](AsyncWebServerRequest *request){
    if(!frameReceived || !frameBuffer){
      request->send(404, "text/plain", "No frame yet");
      return;
    }
    AsyncWebServerResponse *response = request->beginResponse("image/bmp", 10294, [](uint8_t *buffer, size_t maxLen, size_t index) -> size_t {
      size_t bytesSent = 0;
      size_t available = maxLen;
      if (index < 54) {
        size_t toCopy = min((size_t)54 - index, available);
        memcpy(buffer, bmpHeader + index, toCopy);
        index += toCopy; buffer += toCopy; available -= toCopy; bytesSent += toCopy;
      }
      if (available > 0 && index < (54 + 1024)) {
        size_t offset = index - 54;
        size_t toCopy = min((size_t)1024 - offset, available);
        memcpy(buffer, grayscalePalette + offset, toCopy);
        index += toCopy; buffer += toCopy; available -= toCopy; bytesSent += toCopy;
      }
      size_t headerSize = 54 + 1024;
      if (available > 0 && index >= headerSize) {
         size_t dataIdx = index - headerSize; 
         if (dataIdx < 9216 && frameBuffer && frameLen >= 9216) {
             size_t toCopy = min((size_t)9216 - dataIdx, available);
             memcpy(buffer, frameBuffer + dataIdx, toCopy);
             bytesSent += toCopy;
         }
      }
      return bytesSent;
    });
    request->send(response);
  });

  server.begin();
}

void loop() {
  if (frameReadyForInference && !isInferencing) {
      isInferencing = true;
      isInferencing = true;
      frameReadyForInference = false;

      // Copy frameBuffer to input tensor (Preprocessing)
      // This is now done in the loop to keep the network callback fast
      if (frameBuffer && input) {
          for(size_t i=0; i<input->bytes; i++) {
               // Assuming frameBuffer is exactly 96x96 (9216 bytes) at this point
               // and input->bytes is also 9216.
               input->data.int8[i] = (int8_t)((int)frameBuffer[i] - 128); 
          }
      }

      TfLiteStatus invoke_status = interpreter->Invoke();
      
      if (invoke_status == kTfLiteOk) {
          int8_t* results = output->data.int8;
          
          // Softmax Calculation Loop
          float max_score = -1000.0;
          float sum_exp = 0.0;
          int best_idx = -1;
          
          // 1. Find max (for numerical stability) and best index
          for(int i=0; i<kNumClasses; i++){
               float score = (float)results[i]; // Raw int8 score
               lastScores[i] = results[i];      // Save raw for web display
               if(score > max_score){
                   max_score = score;
                   best_idx = i;
               }
          }
          
          // 2. Calculate Exponentials and Sum
          for(int i=0; i<kNumClasses; i++){
              // scale factor 0.125 is typical for int8 models, but can vary
              // For visualization, simple exp is fine if we are consistent
               sum_exp += expf((results[i] - max_score) * 0.125f); 
          }
          
          // 3. Calculate Confidence of Best Match
          lastResult = String(kCategoryLabels[best_idx]);
          float best_exp = expf((results[best_idx] - max_score) * 0.125f);
          lastConfidence = (best_exp / sum_exp) * 100.0f;

          if (lastResult != "Unknown") {
               digitalWrite(LED_PIN, HIGH);
               Serial.println("LED ON - Authorized Face Detected");
          } else {
               digitalWrite(LED_PIN, LOW);
               Serial.println("LED OFF - Unknown Face Detected");
          }

          Serial.printf("Detected: %s (%.1f%%)\n", kCategoryLabels[best_idx], lastConfidence);
      } else {
          lastResult = "Error";
      }

      isInferencing = false; 
  }
}
