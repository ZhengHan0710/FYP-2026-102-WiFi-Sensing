/* ESP32-S3, ESP-IDF 5.4.x. Collect real LLTF CSI; never synthesize samples. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdatomic.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/ip_addr.h"
#include "ping/ping_sock.h"

#if !CONFIG_IDF_TARGET_ESP32S3
#error "Select target esp32s3"
#endif

/* Change these two values before flashing. */
#define HOTSPOT_SSID "CSI_TEST"
#define HOTSPOT_PASSWORD "12345678"
#define PING_INTERVAL_MS 50
#define CONNECTED BIT0
#define DISCONNECTED BIT1
#define MAX_BYTES 256

static const char *TAG = "CSI";
static EventGroupHandle_t events;
static QueueHandle_t frames;
static uint8_t bssid[6];
static atomic_uint received, dropped, ping_ok, ping_timeout;

typedef struct {
    unsigned seq;
    wifi_pkt_rx_ctrl_t rx;
    uint8_t mac[6];
    uint16_t len;
    bool invalid;
    int8_t data[MAX_BYTES];
} frame_t;

static void on_event(void *arg, esp_event_base_t base,
                     int32_t id, void *event_data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        ESP_ERROR_CHECK(esp_wifi_connect());
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *e = event_data;
        xEventGroupClearBits(events, CONNECTED);
        xEventGroupSetBits(events, DISCONNECTED);
        ESP_LOGE(TAG, "DISCONNECTED reason=%d; restart experiment", e->reason);
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(events, CONNECTED);
    }
}

static void on_csi(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || memcmp(info->mac, bssid, 6)) return;
    unsigned seq = atomic_fetch_add(&received, 1);
    if (!info->len || info->len > MAX_BYTES) {
        atomic_fetch_add(&dropped, 1);
        return;
    }
    frame_t f = { .seq = seq, .rx = info->rx_ctrl,
                  .len = info->len, .invalid = info->first_word_invalid };
    memcpy(f.mac, info->mac, 6);
    memcpy(f.data, info->buf, f.len);
    if (xQueueSend(frames, &f, 0) != pdTRUE)
        atomic_fetch_add(&dropped, 1);
}

/* Keep the callback short; serialize outside the Wi-Fi task. */
static void output_task(void *arg)
{
    frame_t f;
    char line[2048];
    TickType_t last_stats = xTaskGetTickCount();
    while (1) {
        if (xQueueReceive(frames, &f, pdMS_TO_TICKS(200)) == pdTRUE) {
            const wifi_pkt_rx_ctrl_t *r = &f.rx;
            /* timestamp, taget_seq, taget are filled on the computer. */
            int n = snprintf(line, sizeof(line),
                "CSI_DATA,%u,,,,%02x:%02x:%02x:%02x:%02x:%02x",
                f.seq, f.mac[0], f.mac[1], f.mac[2],
                f.mac[3], f.mac[4], f.mac[5]);
            int values[] = {
                r->rssi, r->rate, r->sig_mode, r->mcs, r->cwb,
                r->smoothing, r->not_sounding, r->aggregation,
                r->stbc, r->fec_coding, r->sgi, r->noise_floor,
                r->ampdu_cnt, r->channel, r->secondary_channel
            };
            for (unsigned i = 0; i < sizeof(values)/sizeof(values[0]); ++i)
                n += snprintf(line+n, sizeof(line)-n, ",%d", values[i]);
            n += snprintf(line+n, sizeof(line)-n,
                ",%" PRIu32 ",%u,%u,%u,%u,%u,\"[",
                (uint32_t)r->timestamp, (unsigned)r->ant,
                (unsigned)r->sig_len, (unsigned)r->rx_state,
                (unsigned)f.len, (unsigned)f.invalid);
            for (unsigned i = 0; i < f.len; ++i)
                n += snprintf(line+n, sizeof(line)-n,
                              "%s%d", i ? ", " : "", f.data[i]);
            n += snprintf(line+n, sizeof(line)-n, "]\"\n");
            fwrite(line, 1, n, stdout);
        }
        if (xTaskGetTickCount() - last_stats >= pdMS_TO_TICKS(5000)) {
            printf("CSI_STATS,received=%u,dropped=%u,ping_ok=%u,ping_timeout=%u\n",
                atomic_load(&received), atomic_load(&dropped),
                atomic_load(&ping_ok), atomic_load(&ping_timeout));
            last_stats = xTaskGetTickCount();
        }
    }
}

static void on_ping_ok(esp_ping_handle_t h, void *arg)
{ atomic_fetch_add(&ping_ok, 1); }
static void on_ping_timeout(esp_ping_handle_t h, void *arg)
{ atomic_fetch_add(&ping_timeout, 1); }

void app_main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    events = xEventGroupCreate();
    frames = xQueueCreate(64, sizeof(frame_t));
    configASSERT(events && frames);
    configASSERT(xTaskCreate(output_task, "csi_output", 6144, NULL, 4, NULL) == pdPASS);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *netif = esp_netif_create_default_wifi_sta();
    configASSERT(netif);
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    wifi_config_t config = {0};
    snprintf((char *)config.sta.ssid, sizeof(config.sta.ssid), "%s", HOTSPOT_SSID);
    snprintf((char *)config.sta.password, sizeof(config.sta.password), "%s", HOTSPOT_PASSWORD);
    config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    config.sta.pmf_cfg.capable = true;
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &config));
    ESP_ERROR_CHECK(esp_wifi_set_protocol(
    WIFI_IF_STA,
    WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_LOGI(TAG, "Connecting to hotspot...");
    EventBits_t bits = xEventGroupWaitBits(events, CONNECTED | DISCONNECTED,
        pdFALSE, pdFALSE, pdMS_TO_TICKS(30000));
    if (!(bits & CONNECTED) || (bits & DISCONNECTED)) {
        ESP_LOGE(TAG, "Connection failed; check SSID/password/2.4GHz; reset to retry");
        vTaskDelete(NULL);
        return;
    }

    wifi_ap_record_t ap;
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&ap));
    memcpy(bssid, ap.bssid, 6);
    ESP_LOGI(TAG, "AP=%02x:%02x:%02x:%02x:%02x:%02x channel=%u",
        bssid[0], bssid[1], bssid[2], bssid[3], bssid[4], bssid[5], ap.primary);
    wifi_csi_config_t csi = {
        .lltf_en = true, .htltf_en = false, .stbc_htltf2_en = false,
        .ltf_merge_en = false, .channel_filter_en = false,
        .manu_scale = false, .shift = 0
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(on_csi, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    esp_netif_ip_info_t ip;
    ESP_ERROR_CHECK(esp_netif_get_ip_info(netif, &ip));
    ESP_LOGI(TAG, "IP=" IPSTR " gateway=" IPSTR, IP2STR(&ip.ip), IP2STR(&ip.gw));
    esp_ping_config_t ping = ESP_PING_DEFAULT_CONFIG();
    IP_ADDR4(&ping.target_addr, esp_ip4_addr1(&ip.gw), esp_ip4_addr2(&ip.gw),
        esp_ip4_addr3(&ip.gw), esp_ip4_addr4(&ip.gw));
    ping.count = 0;
    ping.interval_ms = PING_INTERVAL_MS;
    ping.timeout_ms = 1000;
    ping.data_size = 32;
    esp_ping_callbacks_t callbacks = {
        .on_ping_success = on_ping_ok, .on_ping_timeout = on_ping_timeout
    };
    esp_ping_handle_t handle;
    ESP_ERROR_CHECK(esp_ping_new_session(&ping, &callbacks, &handle));
    ESP_ERROR_CHECK(esp_ping_start(handle));
    ESP_LOGI(TAG, "READY: real CSI acquisition started");
    xEventGroupWaitBits(events, DISCONNECTED, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_ERROR_CHECK(esp_wifi_set_csi(false));
    ESP_ERROR_CHECK(esp_ping_stop(handle));
    ESP_LOGE(TAG, "Experiment stopped; reset and record again");
    vTaskDelete(NULL);
}
