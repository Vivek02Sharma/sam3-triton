import streamlit as st
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import time
import concurrent.futures

st.title("SAM3 Triton Client")

files = st.file_uploader("Upload Images", type = ["jpg", "jpeg", "png"], accept_multiple_files = True)
prompt = st.text_input("Text Prompt")

if st.button("Process") and prompt and files:
    t_start = time.time()
    batch = files[:32]
    client = httpclient.InferenceServerClient("localhost:9000", connection_timeout = 10, network_timeout = 60, concurrency = 32)

    def infer_image(f):
        try:
            img = np.array(Image.open(f).convert("RGB"))
            t0 = time.time()
            i1 = httpclient.InferInput("RAW_IMAGE", [1, *img.shape], "UINT8")
            i1.set_data_from_numpy(np.expand_dims(img, 0))
            i2 = httpclient.InferInput("TEXT_PROMPT", [1, 1], "BYTES")
            i2.set_data_from_numpy(np.array([[prompt.encode("utf-8")]], dtype=object))
            
            res = client.infer("sam3_model", [i1, i2])
            mask = res.as_numpy("SEGMENTATION_MASK")[0]
            latency = (time.time() - t0) * 1000
            return img, mask, latency, None
        except Exception as e:
            return None, None, None, str(e)

    progress = st.progress(0)
    results = [None] * len(batch)
    with concurrent.futures.ThreadPoolExecutor(max_workers = 32) as executor:
        futures = {executor.submit(infer_image, f): i for i, f in enumerate(batch)}
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results[futures[future]] = future.result()
            progress.progress(done_count / len(batch))

    errors = 0
    for img, mask, latency, err in results:
        if err:
            st.error(f"Failed: {err}")
            errors += 1
            continue
        c1, c2 = st.columns(2)
        c1.image(img, caption = "Original", width = "content")
        c2.image(img * (mask > 0)[..., np.newaxis], caption = f"Extracted ({latency:.0f}ms)", width = "content")

    processed = len(batch) - errors
    st.success(f"Processed {processed}/{len(batch)} images in {(time.time() - t_start) * 1000:.0f}ms total")
