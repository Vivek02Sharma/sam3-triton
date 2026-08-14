# SAM3 Triton Inference Server

High-performance Segment Anything 3 (SAM3) inference using NVIDIA Triton and TensorRT.

## Quickstart

1. **Compile TensorRT engines** (takes a few minutes, requires GPU):
   ```bash
   ./compile.sh
   ```

2. **Start the Triton server**:
   ```bash
   docker compose up -d
   ```
   *Wait until health check passes (`curl -s http://localhost:9000/v2/health/ready`).*

3. **Run the Streamlit UI**:
   ```bash
   # Requires: streamlit, tritonclient[http], opencv-python-headless, transformers, pillow
   streamlit run app.py
   ```

## Design Notes

- **Concurrency**: `app.py` pools 32 HTTP connections. Triton dynamically batches requests.
- **Zero-copy**: `model.py` passes DLPack tensors between the vision/text encoders and mask decoder to skip PCIe transfers.
- **GPU Pinning**: `docker-compose.yml` is hardcoded to GPU 0. Edit `device_ids` if needed.

## References

- Models: [SAM3 ONNX Assets](https://github.com/jamjamjon/assets/releases/tag/sam3)
- Graph Viewer: [Netron](https://netron.app)
