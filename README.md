# Moondream Server
# Moondream: VLM Server and Client

Moondream is a VLM system that provides image understanding capabilities through a server API and client libraries. This repository contains the server implementation and demo client for interacting with the Moondream API.

## Features

- **Image Captioning**: Generate descriptive captions for images
- **Visual Question Answering**: Answer questions about image content
- **Object Detection**: Identify and locate objects within images
- **Point-of-Interest Detection**: Find specific objects in images

## Docker Deployment

### Building the Docker Container

```bash
# Build the Docker image
docker build -t moondream .
```

### Running the Server

```bash
# Run with custom configuration
docker run --rm -it \
  --privileged \
  --cap-add=sys_nice \
  --device=/dev/dri \
  --ipc=host \
  --shm-size=1g \
  --net=host \
  -v $PWD/data:/data \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e PORT=8000 \
  -e =1 \
  -e REQUEST_BATCH_SIZE=32 \
  -e MAX_BATCH_SIZE=16 \
  moondream_server

```

The server will be available at `http://localhost:8000` by default.

#### Persistent Model Cache

To avoid downloading the model each time you start the container, you can mount a local directory to the model cache path:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/model_cache:/tmp/model_cache \
  moondream
```

This will store the downloaded model files in a `model_cache` directory in your current folder.

### Default Docker Configuration

The Dockerfile sets these default environment variables:

```
ONEAPI_DEVICE_SELECTOR="level_zero:0"
TOKENIZERS_PARALLELISM=true
TORCH_COMPILE=true
OMP_NUM_THREADS=28
HOST="0.0.0.0"
PORT=8000
=1
REQUEST_BATCH_SIZE=32
MAX_BATCH_SIZE=16
DEVICE_ID=0
BF16_MODE=true
MODEL_CACHE_DIR=/tmp/model_cache
MOONDREAM_ENABLE_OPTIMIZATION=1
```

You can override any of these values when running the container using the `-e` flag.

### Server Configuration

#### Workers

The `WORKERS` setting controls how many Uvicorn worker processes will be spawned. For XPU-based inference:

- **WORKERS=1** (recommended): Creates a single worker process that loads the model once. This is optimal for GPU workloads as it maximizes available GPU memory for a single model instance.
- **WORKERS>1**: Creates multiple worker processes, each loading its own copy of the model. This is generally not recommended for XPU workloads as it divides GPU memory between multiple model instances and can lead to out-of-memory errors and compute starving.

Since the server uses FastAPI with async endpoints, a single worker can efficiently handle multiple concurrent requests using asyncio's event loop, even with `WORKERS=1`.

#### IPEX Optimization Toggle

The `MOONDREAM_ENABLE_OPTIMIZATION` environment variable controls whether IPEX optimizations are applied to the model:

- **MOONDREAM_ENABLE_OPTIMIZATION=1** (default): Enables IPEX optimizations for better performance on Intel GPUs.
- **MOONDREAM_ENABLE_OPTIMIZATION=0**: Disables IPEX optimizations, using the model as-is.

This toggle is useful for benchmarking and comparing performance with and without optimizations. Accepted values are `1`, `true`, `yes`, `on` for enabling, and `0`, `false`, `no`, `off` for disabling (case-insensitive).


## Using the Demo Client

The repository includes a demo client (`demo_client.py`) that demonstrates how to interact with the Moondream API.

### Installation

```bash
# Install required dependencies
pip install requests aiohttp
```

### Basic Usage

```bash
# Run the demo client with default settings
python demo_client.py

# Specify a custom server URL
python demo_client.py --url http://localhost:8000
```

### Example Operations

#### Image Captioning

```python
# Generate a caption for an image
caption = caption_image(url, image_path, length="normal")
print(f"Caption: {caption}")
```

#### Visual Question Answering

```python
# Ask a question about an image
answer = query_image(url, image_path, "What can you see in this image?")
print(f"Answer: {answer}")
```

#### Object Detection

```python
# Detect objects in an image
objects = detect_objects(url, image_path, "person")
print(f"Detected {len(objects)} objects")
for obj in objects:
    print(f"- {obj['label']} (confidence: {obj['confidence']:.2f})")
```

#### Smart Object Detection

```python
# First caption the image, then detect relevant objects
smart_object_detection(url, image_path)
```

#### Batch Processing

```python
# Process multiple images with different operations
results = batch_process(url, 
                       [image_path1, image_path2], 
                       [{"op": "caption"}, {"op": "query", "prompt": "What is this?"}])
```

#### Asynchronous Processing

```python
# Process images asynchronously
job_id = batch_process_async(url, [image_path], [{"op": "caption"}])
results = wait_for_batch_completion(url, job_id)
```

## Environment Variables

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `WORKERS` | `1` | Number of FastAPI worker processes |
| `GRACEFUL_TIMEOUT` | `120` | Graceful shutdown timeout (seconds) |
| `KEEP_ALIVE` | `120` | Keep-alive timeout (seconds) |
| `MAX_REQUESTS` | `10000` | Maximum requests per worker |
| `BACKLOG` | `2048` | Connection queue size |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `vikhyatk/moondream2` | Model ID from Hugging Face |
| `REVISION` | `None` | Model revision (None = main branch) |
| `DEVICE_ID` | `0` | Intel GPU device ID |
| `BF16_MODE` | `true` | Whether to use BF16 precision (falls back to FP32 if BF16 not supported) |
| `MODEL_CACHE_DIR` | `None` | Directory to cache model files |
| `OPTIMIZATION_LEVEL` | `O1` | IPEX optimization level (O0, O1, O2) - O1 includes conv+bn folding, weights prepack, dropout removal |

### Batch Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUEST_BATCH_SIZE` | `16` | Maximum requests in a batch |
| `MAX_BATCH_SIZE` | `8` | Maximum model inference batch size |
| `MAX_CONCURRENCY` | `4` | Number of parallel processing threads |
| `BATCH_TIMEOUT` | `0.1` | Seconds to wait for batching |
| `WORKER_TIMEOUT` | `300` | Worker timeout (seconds) |
| `BATCH_JOB_MAX_AGE` | `3600` | Maximum age for batch jobs (seconds) |
| `COMPLETED_JOB_RETENTION` | `600` | Retention period for completed jobs (seconds) |
| `MIN_JOB_RETENTION` | `30` | Minimum job retention period (seconds) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/caption` | POST | Generate image caption |
| `/query` | POST | Answer question about image |
| `/detect` | POST | Detect objects in image |
| `/point` | POST | Find points of interest |
| `/batch` | POST | Process multiple operations |
| `/batch/async` | POST | Process batch asynchronously |
| `/batch/status/{job_id}` | GET | Check async job status |
