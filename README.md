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
# Run the container with default settings
docker run -p 8000:8000 moondream

# Run with custom configuration
docker run -p 8000:8000 \
  -e PORT=8000 \
  -e WORKERS=16 \
  -e REQUEST_BATCH_SIZE=32 \
  -e MAX_BATCH_SIZE=16 \
  moondream

# Run with persistent cache directories
docker run -p 8000:8000 \
  -v $(pwd)/model_cache:/tmp/model_cache \
  -v $(pwd)/ipex_cache:/tmp/ipex_cache \
  -v $(pwd)/torch_cache:/tmp/torch_cache \
  moondream
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
WORKERS=64
REQUEST_BATCH_SIZE=32
MAX_BATCH_SIZE=16
DEVICE_ID=0
BF16_MODE=true
MODEL_CACHE_DIR=/tmp/model_cache
```

You can override any of these values when running the container using the `-e` flag.

## Server Capabilities

### Processing Modes

#### Synchronous Processing
- **Best for**: Interactive applications, immediate responses, simple requests
- **Endpoints**: `/caption`, `/query`, `/detect`, `/point`, `/batch`
- **Behavior**: Blocks until processing completes, returns results directly
- **Use when**: Response times are expected to be short (<5 seconds) or when client needs immediate results

#### Asynchronous Processing
- **Best for**: Long-running tasks, large batches, background processing
- **Endpoints**: `/batch/async`
- **Behavior**: Returns a job ID immediately, client can poll for results using `/batch/status/{job_id}`
- **Use when**: Processing multiple images, running complex operations, or when response times might exceed client timeouts

### Features

#### Adaptive Batching
- Automatically groups similar requests to optimize GPU hits (sequential op with request batching)
- Dynamically adjusts batch sizes based on available resources
- Reduces latency by processing multiple requests in parallel when possible
- Configurable via `REQUEST_BATCH_SIZE`, `MAX_BATCH_SIZE`, and `BATCH_TIMEOUT` environment variables

#### Resource Management
-  Memory management to prevent OOM errors
- Automatic cleanup of completed jobs based on configurable retention policies
- Graceful degradation under heavy load
- Health monitoring via the `/health` endpoint

#### High-Performance Request Handling
-  In-memory queuing system
- Concurrent request processing with thread pooling
- Efficient request batching with dynamic timeout management
- Optimized for high-throughput / low latency scenarios with minimal overhead
- Robust error handling and recovery mechanisms

### Performance Optimization

#### For Low Latency (Faster Response Times)
- Use individual endpoints (`/caption`, `/query`, `/detect`, `/point`) for single operations
- Set `BATCH_TIMEOUT=0.0` to process requests immediately without waiting for batching
- Reduce `MAX_BATCH_SIZE` to 1-4 for quicker processing of individual requests
- Increase `WORKERS` for more concurrent request handling

#### For High Throughput (Maximum Processing Volume)
- Use `/batch` for synchronous batch processing of multiple images
- Use `/batch/async` for very large batches or background processing
- Increase `REQUEST_BATCH_SIZE` and `MAX_BATCH_SIZE` (16-32) to process more images per batch
- Set `BATCH_TIMEOUT=0.1` to collect more requests into batches

#### Balanced Approach
- Use default settings which balance latency and throughput
- Adjust `MAX_CONCURRENCY` to match your hardware capabilities
- Monitor the `/health` endpoint to track resource utilization

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
| `WORKERS` | `16` | Number of FastAPI worker processes |
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
