# DA3408 A4 — CPU Image Captioning with Prometheus Monitoring

A CPU-only image captioning app (BLIP) built with Streamlit, instrumented with `prometheus_client`, and monitored with Prometheus, Alertmanager and node_exporter.

- **Single mode:** upload one image, get a caption.
- **Bulk mode:** upload a ZIP of images (max 100 images, 300 MB uncompressed), get a table of captions and a CSV download.
- **Observability:** application metrics on a separate port (`:8000/metrics`), host metrics via node_exporter, and alert rules routed to Alertmanager.

---

## Architecture

```
 Browser ──► Streamlit app (:8501) ──► /metrics (:8000) ─┐
                                                         │ scrape every 5 s
 node_exporter (:9100) ──────────────────────────────────┼──► Prometheus (:9090)
                                                         │          │
                                                         │          └─ alert rules ──► Alertmanager (:9093)
```

| Component | Port | Runs as |
|---|---|---|
| Streamlit app | 8501 | Python process on the host |
| App metrics (`prometheus_client`) | 8000 | Same process, separate port |
| node_exporter | 9100 | Host service (`prometheus-node-exporter`) |
| Prometheus | 9090 | Docker (host network) |
| Alertmanager | 9093 | Docker (host network) |

---

## Repository layout

```
.
├── app.py                      # Streamlit app + all Prometheus instrumentation
├── requirements.txt
├── prometheus.yml              # scrape configs (app, node_exporter, prometheus) + alerting
├── alerts.yml                  # alert rules
├── alertmanager.yml            # minimal Alertmanager config
├── docker-compose.yml          # Prometheus + Alertmanager (+ Grafana, dashboard not covered here)
├── test_images.zip             # sample images for testing Bulk mode
├── AI_DISCLOSURE.md            # how AI was used in this project
└── README.md
```

---

## Quick start (Linux)

> `docker-compose.yml` uses `network_mode: host`, which works on Linux only.

### 1. Install node_exporter

```bash
sudo apt install prometheus-node-exporter
systemctl status prometheus-node-exporter    # should be active, listening on :9100
```

### 2. Install and run the app

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the BLIP model (~1 GB). Then:

- App UI: <http://localhost:8501>
- Raw metrics: <http://localhost:8000/metrics>

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `Salesforce/blip-image-captioning-base` | Hugging Face captioning model |
| `METRICS_PORT` | `8000` | Port for `/metrics` |
| `TORCH_THREADS` | `4` | CPU threads for inference |

### 3. Start the monitoring stack

```bash
docker compose up -d
```

- Prometheus: <http://localhost:9090> (check **Status → Targets**: `captioner_app`, `node`, `prometheus` should all be UP)
- Alertmanager: <http://localhost:9093>

---

## Metrics catalogue

All application metrics are prefixed `captioner_`. Counters are exposed with a `_total` suffix; Histograms and Summaries expose `_sum` / `_count` (Histograms also `_bucket`).

### Counters

| Metric | Labels | Meaning |
|---|---|---|
| `captioner_images_processed_total` | `mode`, `status`, `model` | Images processed, by single/bulk and success/error |
| `captioner_client_requests_total` | `mode`, `client_ip`, `session_id` | Requests per client |
| `captioner_client_images_total` | `mode`, `client_ip`, `session_id` | Images submitted per client |
| `captioner_errors_total` | `mode`, `error_type` | Errors by cause (`too_large`, `decode_error`, `inference_error`, `other`) |
| `captioner_zip_uploads_total` | `status` | ZIP outcomes (`ok`, `invalid`, `too_large`, `no_images`) |

### Gauges

| Metric | Labels | Meaning |
|---|---|---|
| `captioner_active_requests` | `mode` | Requests currently being handled |
| `captioner_inference_queue_depth` | — | Images waiting for the model lock |
| `captioner_bulk_images_remaining` | — | Images left in the current ZIP job |
| `captioner_model_size_bytes` | `model` | Model weights size in memory |
| `captioner_model_load_seconds` | `model` | Model load time |
| `captioner_process_memory_bytes` | `type` (`rss`, `vms`) | App process memory |
| `captioner_process_cpu_percent` | — | App process CPU (100 = one core) |

### Histograms

| Metric | Labels | Meaning |
|---|---|---|
| `captioner_inference_latency_seconds` | `mode` | Model inference time per image |
| `captioner_queue_wait_seconds` | — | Time waiting for the model lock |
| `captioner_image_size_bytes` | `mode` | Uploaded image size |
| `captioner_zip_images_per_upload` | — | Images per ZIP |
| `captioner_zip_processing_seconds` | — | Total time to process a ZIP |

### Summaries (`_sum` and `_count`)

| Metric | Labels | Meaning |
|---|---|---|
| `captioner_request_duration_seconds` | `mode` | End-to-end request duration |
| `captioner_caption_length_chars` | `mode` | Caption length in characters |
| `captioner_caption_length_words` | `mode` | Caption length in words |

### Info

| Metric | Labels | Meaning |
|---|---|---|
| `captioner_app_info` | `model`, `torch`, `device`, `torch_threads` | Static application info |

### Custom labels and cardinality

`client_ip` and `session_id` identify clients (taken from `X-Forwarded-For` / the Streamlit session). They are high-cardinality, so they are used only on the two per-client counters, and sessions are expected to be short-lived. All other metrics use low-cardinality labels (`mode`, `status`, `error_type`).

### Host metrics

node_exporter provides CPU (`node_cpu_seconds_total`), memory (`node_memory_*`), disk (`node_filesystem_*`, `node_disk_*`) and load metrics, scraped by the `node` job.

---

## Alert rules

Defined in `alerts.yml`, evaluated every 5 s, sent to Alertmanager.

| Alert | Condition | For | Severity |
|---|---|---|---|
| `CaptionerAppDown` | `up{job="captioner_app"} == 0` | 15 s | critical |
| `HighInferenceLatencyP95` | p95 inference latency > 10 s | 30 s | warning |
| `HighImageErrorRatio` | error ratio > 10 % | 30 s | warning |
| `HighHostCPU` | host CPU > 85 % | 30 s | warning |
| `HighHostMemory` | host memory > 90 % | 30 s | warning |

---

---

## Generating test load

1. Start the app and the monitoring stack (steps above).
2. Upload 3–5 single images (e.g. from `test_images.zip`).
3. Upload `test_images.zip` in the Bulk tab (or any ZIP of 30–50 images) and watch the metrics change in Prometheus (<http://localhost:9090>) or at `:8000/metrics`.
4. Trigger some errors: a corrupt file renamed to `.jpg`, an empty ZIP, and a non-ZIP file renamed to `.zip`.
5. Optionally open a second browser window to produce a second client session.

---

## Design notes

- **Metrics created once:** Streamlit re-runs the script on every interaction, so metrics and the metrics server are created inside `st.cache_resource`.
- **Single model lock:** inference is serialised with a lock, and waiting time is exposed as `inference_queue_depth` and `queue_wait_seconds`.
- **Pre-created label sets:** key label combinations are initialised at start-up so panels and alerts show `0` rather than "no data".
- **Safety limits:** max 100 images per ZIP, 15 MB per image, 300 MB uncompressed (zip-bomb guard), and images are downscaled to at most 1024 px before inference.

---