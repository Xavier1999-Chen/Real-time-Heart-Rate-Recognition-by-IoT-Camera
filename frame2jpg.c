#include <stdio.h>
#include <string.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_camera.h>
#include <esp_log.h>
#include <mqtt_client.h>
#include <esp_wifi.h>
#include <esp_event_loop.h>

// 定义MQTT服务器信息
#define MQTT_HOST "mqtt.example.com"
#define MQTT_PORT 1883
#define MQTT_USERNAME "mqtt_user"
#define MQTT_PASSWORD "mqtt_password"
#define MQTT_TOPIC "image_topic"

// 定义摄像头分辨率和JPEG压缩质量
#define CAMERA_WIDTH 640
#define CAMERA_HEIGHT 480
#define JPEG_QUALITY 10

// 定义MQTT客户端实例
esp_mqtt_client_handle_t mqtt_client;

// 定义摄像头初始化函数
void camera_init() {
    // 配置摄像头驱动
    camera_config_t config = {
        .pin_pwdn = -1,
        .pin_reset = -1,
        .pin_xclk = 4,
        .pin_sscb_sda = 18,
        .pin_sscb_scl = 23,
        .pin_d7 = 36,
        .pin_d6 = 37,
        .pin_d5 = 38,
        .pin_d4 = 39,
        .pin_d3 = 35,
        .pin_d2 = 34,
        .pin_d1 = 33,
        .pin_d0 = 32,
        .pin_vsync = 5,
        .pin_href = 27,
        .pin_pclk = 25,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_SVGA,
        .jpeg_quality = JPEG_QUALITY,
        .fb_count = 1
    };
    // 初始化摄像头驱动
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE("Camera", "Failed to initialize camera: %s", esp_err_to_name(err));
    }
}

// 定义MQTT事件处理函数
esp_err_t mqtt_event_handler(esp_mqtt_event_handle_t event) {
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI("MQTT", "Connected to MQTT server");
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI("MQTT", "Disconnected from MQTT server");
            break;
        default:
            break;
    }
    return ESP_OK;
}

// 定义任务函数，用于读取摄像头图像并发送到MQTT服务器
void camera_task(void *pvParameters) {
    while (1) {
        // 从摄像头读取图像
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE("Camera", "Failed to acquire camera frame buffer");
            continue;
        }
        // 压缩图像为JPEG格式
        size_t out_len;
        uint8_t *out_buf;
        bool ok = frame2jpg(fb, JPEG_QUALITY, &out_buf,&out_len);
        if (!ok) {
            ESP_LOGE("Camera", "Failed to compress camera frame");
            esp_camera_fb_return(fb);
            continue;
        }
        // 创建MQTT消息
        char topic[64];
        sprintf(topic, "%s", MQTT_TOPIC);
        esp_mqtt_client_publish(mqtt_client, topic, (char *)out_buf, out_len, 0, 0);
        ESP_LOGI("MQTT", "Sent image of size %d", out_len);
        // 释放图像数据和缓冲区
        free(out_buf);
        esp_camera_fb_return(fb);
        // 延迟一段时间
        vTaskDelay(500 / portTICK_PERIOD_MS);
    }
}

// 定义主函数
void app_main() {
    // 初始化摄像头
    camera_init();
    // 初始化WiFi连接
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_wifi_set_mode(WIFI_MODE_STA);
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "wifi_ssid",
            .password = "wifi_password"
        }
    };
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
    // 初始化MQTT客户端
    esp_mqtt_client_config_t mqtt_cfg = {
        .uri = MQTT_HOST,
        .port = MQTT_PORT,
        .username = MQTT_USERNAME,
        .password = MQTT_PASSWORD,
        .event_handle = mqtt_event_handler
    };
    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_start(mqtt_client);
    // 创建任务，用于读取摄像头图像并发送到MQTT服务器
    xTaskCreate(&camera_task, "camera_task", 4096, NULL, 5, NULL);
}