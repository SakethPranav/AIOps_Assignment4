"""
DA3408 A4 - CPU-only image captioning with Prometheus instrumentation.

Run:      streamlit run app.py
Metrics:  http://localhost:8000/metrics   (separate port from Streamlit's 8501)
"""
import csv
import io
import os
import threading
import time
import zipfile

import psutil
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram, Info, Summary, start_http_server
from transformers import BlipForConditionalGeneration, BlipProcessor

# ----------------------------------------------------------------------------
# Config (override with environment variables)
# ----------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "Salesforce/blip-image-captioning-base")
MODEL_NAME = MODEL_ID.split("/")[-1]
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "4"))   # leave cores free for the OS / monitoring

ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MAX_ZIP_IMAGES = 100                     # cap per upload so a CPU run can't run away
MAX_IMAGE_BYTES = 15 * 1024 * 1024       # 15 MB per image
MAX_ZIP_UNCOMPRESSED = 300 * 1024 * 1024  # zip-bomb guard (300 MB of images)


# ----------------------------------------------------------------------------
# Metrics. Streamlit re-runs this script on every interaction, so metrics must be
# created exactly once per process -> cache_resource.
# ----------------------------------------------------------------------------
@st.cache_resource
def init_metrics():
    m = {}
    p = "captioner_"

    # ---- Counters ----
    m["images"] = Counter(p + "images_processed", "Images processed, by mode and outcome",
                          ["mode", "status", "model"])
    m["client_requests"] = Counter(p + "client_requests", "Requests per client (cardinality: keep sessions short-lived)",
                                   ["mode", "client_ip", "session_id"])
    m["client_images"] = Counter(p + "client_images", "Images submitted per client",
                                 ["mode", "client_ip", "session_id"])
    m["errors"] = Counter(p + "errors", "Errors by type", ["mode", "error_type"])
    m["zips"] = Counter(p + "zip_uploads", "ZIP uploads by outcome", ["status"])

    # ---- Gauges ----
    m["active"] = Gauge(p + "active_requests", "Requests currently being handled", ["mode"])
    m["queue_depth"] = Gauge(p + "inference_queue_depth", "Requests waiting for the model lock")
    m["bulk_remaining"] = Gauge(p + "bulk_images_remaining", "Images left in the current bulk job")
    m["model_size"] = Gauge(p + "model_size_bytes", "Size of model weights in memory", ["model"])
    m["model_load"] = Gauge(p + "model_load_seconds", "Time taken to load the model", ["model"])
    mem = Gauge(p + "process_memory_bytes", "App process memory", ["type"])
    proc = psutil.Process()
    mem.labels(type="rss").set_function(lambda: proc.memory_info().rss)
    mem.labels(type="vms").set_function(lambda: proc.memory_info().vms)
    cpu = Gauge(p + "process_cpu_percent", "App process CPU % (100 = one full core)")
    proc.cpu_percent(None)  # prime the counter
    cpu.set_function(lambda: proc.cpu_percent(None))

    # ---- Histograms ----
    m["latency"] = Histogram(p + "inference_latency_seconds", "Model inference time per image", ["mode"],
                             buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60))
    m["queue_wait"] = Histogram(p + "queue_wait_seconds", "Time spent waiting for the model lock",
                                buckets=(0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60))
    m["img_bytes"] = Histogram(p + "image_size_bytes", "Uploaded image size", ["mode"],
                               buckets=(1e4, 5e4, 1e5, 5e5, 1e6, 2e6, 5e6, 1.5e7))
    m["zip_images"] = Histogram(p + "zip_images_per_upload", "Images per ZIP upload",
                                buckets=(1, 5, 10, 20, 40, 60, 80, 100))
    m["zip_seconds"] = Histogram(p + "zip_processing_seconds", "Total time to process a ZIP",
                                 buckets=(1, 5, 10, 30, 60, 120, 300, 600))

    # ---- Summaries (exposed as _count and _sum; the Python client has no quantiles) ----
    m["req_seconds"] = Summary(p + "request_duration_seconds", "End-to-end request duration", ["mode"])
    m["cap_chars"] = Summary(p + "caption_length_chars", "Caption length in characters", ["mode"])
    m["cap_words"] = Summary(p + "caption_length_words", "Caption length in words", ["mode"])

    info = Info(p + "app", "Static application info")
    info.info({"model": MODEL_ID, "torch": torch.__version__, "device": "cpu",
               "torch_threads": str(TORCH_THREADS)})

    # Pre-create label sets so panels/alerts show 0 instead of "no data" before first use.
    for mode in ("single", "bulk"):
        for status in ("success", "error"):
            m["images"].labels(mode=mode, status=status, model=MODEL_NAME)
        m["active"].labels(mode=mode)
        m["latency"].labels(mode=mode)
        m["req_seconds"].labels(mode=mode)
    for status in ("ok", "invalid", "too_large", "no_images"):
        m["zips"].labels(status=status)

    start_http_server(METRICS_PORT)
    return m


@st.cache_resource
def load_model(_m):
    torch.set_num_threads(TORCH_THREADS)
    t0 = time.perf_counter()
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).eval()
    _m["model_load"].labels(model=MODEL_NAME).set(time.perf_counter() - t0)
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    _m["model_size"].labels(model=MODEL_NAME).set(size)
    return processor, model, threading.Lock()


M = init_metrics()
with st.spinner("Loading model (first run downloads ~1 GB)..."):
    processor, model, model_lock = load_model(M)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def client_info():
    """Best-effort client metadata used as metric labels."""
    ip = "local"
    try:
        fwd = st.context.headers.get("X-Forwarded-For", "")
        ip = fwd.split(",")[0].strip() or st.context.ip_address or "local"
    except Exception:
        pass
    sid = "unknown"
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        sid = get_script_run_ctx().session_id[:8]
    except Exception:
        pass
    return ip, sid


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))  # BLIP resizes to 384 anyway; this caps memory
    return img


def generate_caption(img: Image.Image, mode: str) -> str:
    """Runs BLIP on CPU. Serialised with a lock; waiting time is tracked as queue depth/wait."""
    M["queue_depth"].inc()
    t_wait = time.perf_counter()
    with model_lock:
        M["queue_depth"].dec()
        M["queue_wait"].observe(time.perf_counter() - t_wait)
        t0 = time.perf_counter()
        inputs = processor(images=img, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=30, num_beams=3)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        M["latency"].labels(mode=mode).observe(time.perf_counter() - t0)
    M["cap_chars"].labels(mode=mode).observe(len(caption))
    M["cap_words"].labels(mode=mode).observe(len(caption.split()))
    return caption


def caption_bytes(data: bytes, mode: str, ip: str, sid: str):
    """Caption one image's bytes with full metric accounting. Returns (caption, image, error)."""
    M["client_images"].labels(mode=mode, client_ip=ip, session_id=sid).inc()
    M["img_bytes"].labels(mode=mode).observe(len(data))
    try:
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("too_large")
        try:
            img = load_image(data)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError("decode_error")
        try:
            cap = generate_caption(img, mode)
        except Exception:
            raise ValueError("inference_error")
        M["images"].labels(mode=mode, status="success", model=MODEL_NAME).inc()
        return cap, img, None
    except ValueError as e:
        kind = str(e) if str(e) in ("too_large", "decode_error", "inference_error") else "other"
        M["errors"].labels(mode=mode, error_type=kind).inc()
        M["images"].labels(mode=mode, status="error", model=MODEL_NAME).inc()
        return None, None, kind


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="CPU Image Captioning", page_icon="🖼️", layout="centered")
st.title("🖼️ CPU Image Captioning")
st.caption(f"Model: `{MODEL_ID}` · CPU only · metrics at `http://localhost:{METRICS_PORT}/metrics`")

ip, sid = client_info()
tab_single, tab_bulk = st.tabs(["Single image", "Bulk (ZIP)"])

# ---- Single mode ----
with tab_single:
    up = st.file_uploader("Upload an image", type=[e.strip(".") for e in ALLOWED_EXT], key="single_up")
    if up and st.button("Generate caption", type="primary", key="single_go"):
        M["client_requests"].labels(mode="single", client_ip=ip, session_id=sid).inc()
        with M["active"].labels(mode="single").track_inprogress(), \
                M["req_seconds"].labels(mode="single").time():
            with st.spinner("Captioning..."):
                cap, img, err = caption_bytes(up.getvalue(), "single", ip, sid)
        if err:
            st.error(f"Could not process this image ({err}).")
        else:
            st.image(img, use_container_width=True)
            st.success(cap)

# ---- Bulk mode ----
with tab_bulk:
    zup = st.file_uploader("Upload a ZIP of images", type=["zip"], key="bulk_up")
    if zup and st.button("Caption all images", type="primary", key="bulk_go"):
        M["client_requests"].labels(mode="bulk", client_ip=ip, session_id=sid).inc()
        rows = []
        t_start = time.perf_counter()
        with M["active"].labels(mode="bulk").track_inprogress(), \
                M["req_seconds"].labels(mode="bulk").time():
            try:
                zf = zipfile.ZipFile(io.BytesIO(zup.getvalue()))
                infos = [i for i in zf.infolist()
                         if not i.is_dir()
                         and i.filename.lower().endswith(ALLOWED_EXT)
                         and not i.filename.startswith("__MACOSX")
                         and not os.path.basename(i.filename).startswith(".")]
            except zipfile.BadZipFile:
                M["zips"].labels(status="invalid").inc()
                st.error("That doesn't look like a valid ZIP file.")
                infos = None

            if infos is not None:
                if not infos:
                    M["zips"].labels(status="no_images").inc()
                    st.warning("No supported images found in the ZIP.")
                elif sum(i.file_size for i in infos) > MAX_ZIP_UNCOMPRESSED:
                    M["zips"].labels(status="too_large").inc()
                    st.error("ZIP is too large when uncompressed (limit 300 MB).")
                else:
                    if len(infos) > MAX_ZIP_IMAGES:
                        st.warning(f"Only the first {MAX_ZIP_IMAGES} of {len(infos)} images will be processed.")
                        infos = infos[:MAX_ZIP_IMAGES]
                    M["zips"].labels(status="ok").inc()
                    M["zip_images"].observe(len(infos))
                    bar = st.progress(0.0, text="Starting...")
                    try:
                        for n, info in enumerate(infos, 1):
                            M["bulk_remaining"].set(len(infos) - n + 1)
                            t0 = time.perf_counter()
                            data = zf.read(info) if info.file_size <= MAX_IMAGE_BYTES else b"x" * (MAX_IMAGE_BYTES + 1)
                            cap, _, err = caption_bytes(data, "bulk", ip, sid)
                            rows.append({"file": info.filename, "caption": cap or "",
                                         "status": "ok" if not err else err,
                                         "seconds": round(time.perf_counter() - t0, 2)})
                            bar.progress(n / len(infos), text=f"{n}/{len(infos)} · {info.filename}")
                    finally:
                        M["bulk_remaining"].set(0)
                    M["zip_seconds"].observe(time.perf_counter() - t_start)
                    st.session_state["bulk_rows"] = rows

    rows = st.session_state.get("bulk_rows")
    if rows:
        st.dataframe(rows, use_container_width=True)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        st.download_button("Download captions (CSV)", buf.getvalue(), "captions.csv", "text/csv")
