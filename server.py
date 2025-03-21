import asyncio
import gc
import json
import multiprocessing
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import intel_extension_for_pytorch as ipex  # noqa
import psutil
import torch
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from backend import MoondreamEngine

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",  # change to INFO or WARNING if too many logs
)

app = FastAPI(
    title="Moondream API",
    description="Vision-Language API powered by Moondream 2 on Intel XPU",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_ID = os.getenv("MODEL_ID", "vikhyatk/moondream2")
REVISION = os.getenv("REVISION", None)
DEVICE_ID = int(os.getenv("DEVICE_ID", "0"))
BF16_MODE = os.getenv("BF16_MODE", "true").lower() in ["true", "1", "yes"]
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/root/.cache/huggingface")

REQUEST_BATCH_SIZE = int(
    os.getenv("REQUEST_BATCH_SIZE", "16")
)  # Maximum requests to collect before processing
MAX_BATCH_SIZE = int(
    os.getenv("MAX_BATCH_SIZE", "8")
)  # Maximum batch size for model inference
WORKERS = int(
    os.getenv("WORKERS", 1)
)  # number of workers that the fastapi server should use
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))  # Parallel processing threads
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.1"))  # Seconds to wait for batching
WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT", "300"))  # Worker timeout in seconds
KEEP_ALIVE = int(os.getenv("KEEP_ALIVE", "120"))  # Keep-alive timeout in seconds
MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", "10000"))  # Max requests per worker
BACKLOG = int(os.getenv("BACKLOG", "2048"))  # Connection queue size
GRACEFUL_TIMEOUT = int(os.getenv("GRACEFUL_TIMEOUT", "120"))

# Maximum age for batch jobs in seconds (default: 1 hour)
BATCH_JOB_MAX_AGE = int(os.environ.get("BATCH_JOB_MAX_AGE", 3600))
# Retention period for completed jobs in seconds (default: 10 minutes)
COMPLETED_JOB_RETENTION = int(os.environ.get("COMPLETED_JOB_RETENTION", 600))
# Minimum retention period to ensure clients can fetch results (default: 30 seconds)
MIN_JOB_RETENTION = int(os.environ.get("MIN_JOB_RETENTION", 30))


engine: Optional[MoondreamEngine] = None
executor: Optional[ThreadPoolExecutor] = None
batch_semaphore: Optional[asyncio.Semaphore] = None
request_queues = {}
queue_locks = {}
processing_tasks = {}


class QueryRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to process with the image")


class DetectRequest(BaseModel):
    object_type: str = Field(..., description="Type of object to detect")


class PointRequest(BaseModel):
    object_type: str = Field(..., description="Type of object to point to")


class BatchJob(BaseModel):
    job_id: str
    status: str
    created_at: float
    completed_at: Optional[float] = None
    results: Optional[List[Dict]] = None
    error: Optional[str] = None


batch_jobs = {}


async def cleanup_old_batch_jobs():
    """Background task to clean up old batch jobs to prevent memory leaks"""
    while True:
        try:
            current_time = time.time()
            jobs_to_remove = []
            status_counts = {}
            for job in batch_jobs.values():
                status = job.status
                if status not in status_counts:
                    status_counts[status] = 0
                status_counts[status] += 1
            if batch_jobs:
                logger.debug(f"Current job status distribution: {status_counts}")
            for job_id, job in batch_jobs.items():
                if (
                    job.status in ["completed", "failed"]
                    and hasattr(job, "completed_at")
                    and job.completed_at
                ):
                    time_since_completion = current_time - job.completed_at
                    if (
                        time_since_completion > COMPLETED_JOB_RETENTION
                        and time_since_completion > MIN_JOB_RETENTION
                    ):
                        expiry_time = job.completed_at + COMPLETED_JOB_RETENTION
                        retention_period = COMPLETED_JOB_RETENTION
                        jobs_to_remove.append(
                            {
                                "job_id": job_id,
                                "status": job.status,
                                "age": current_time - job.created_at,
                                "time_since_completion": time_since_completion,
                                "expiry_time": time.ctime(expiry_time),
                                "retention_period": f"{retention_period}s after completion",
                            }
                        )
                else:
                    job_age = current_time - job.created_at
                    if job_age > BATCH_JOB_MAX_AGE:
                        expiry_time = job.created_at + BATCH_JOB_MAX_AGE
                        retention_period = BATCH_JOB_MAX_AGE
                        jobs_to_remove.append(
                            {
                                "job_id": job_id,
                                "status": job.status,
                                "age": job_age,
                                "expiry_time": time.ctime(expiry_time),
                                "retention_period": f"{BATCH_JOB_MAX_AGE}s from creation",
                            }
                        )
            logger.info(
                f"Before cleanup, current batch jobs: {list(batch_jobs.keys())}"
            )
            logger.info(
                f"Jobs scheduled for removal: {[j['job_id'] for j in jobs_to_remove]}"
            )
            for job_info in jobs_to_remove:
                job_id = job_info["job_id"]
                status = job_info["status"]
                if job_id in batch_jobs:
                    job = batch_jobs[job_id]
                    completed_at = getattr(job, "completed_at", None)
                    if status in ["completed", "failed"] and completed_at:
                        time_since_completion = time.time() - completed_at
                        if time_since_completion < MIN_JOB_RETENTION:
                            logger.warning(
                                f"Skipping removal of job {job_id} because it was completed only {time_since_completion:.1f}s ago "
                                f"(MIN_JOB_RETENTION is {MIN_JOB_RETENTION}s)"
                            )
                            continue
                    if "time_since_completion" in job_info:
                        logger.info(
                            f"Removing {status} batch job: {job_id} (age: {job_info['age']:.1f}s, time since completion: {job_info['time_since_completion']:.1f}s)"
                        )
                        logger.info(
                            f"Job {job_id} expired at: {job_info['expiry_time']} (retention: {job_info['retention_period']})"
                        )
                    else:
                        logger.info(
                            f"Removing {status} batch job: {job_id} (age: {job_info['age']:.1f}s)"
                        )
                        logger.info(
                            f"Job {job_id} expired at: {job_info['expiry_time']} (retention: {job_info['retention_period']})"
                        )
                    job_age = time.time() - job.created_at
                    if job_age < MIN_JOB_RETENTION:
                        logger.warning(
                            f"Skipping removal of job {job_id} because it's only {job_age:.1f}s old "
                            f"(MIN_JOB_RETENTION is {MIN_JOB_RETENTION}s)"
                        )
                        continue
                    batch_jobs.pop(job_id, None)
                    logger.info(f"Successfully removed job {job_id}")
            if jobs_to_remove:
                logger.info(
                    f"Cleaned up {len(jobs_to_remove)} expired batch jobs. Current job count: {len(batch_jobs)}"
                )
                remaining_status = {}
                for job in batch_jobs.values():
                    if job.status not in remaining_status:
                        remaining_status[job.status] = 0
                    remaining_status[job.status] += 1
                if remaining_status:
                    logger.info(f"Remaining jobs by status: {remaining_status}")
        except Exception as e:
            logger.error(f"Error in batch job cleanup: {str(e)}")
        await asyncio.sleep(300)


@app.on_event("startup")
async def startup_event():
    global engine, executor, batch_semaphore, request_queues, queue_locks, processing_tasks

    try:
        logger.info(f"Initializing Moondream Engine (model: {MODEL_ID})")
        start_time = time.time()
        engine = MoondreamEngine(
            model_id=MODEL_ID,
            revision=REVISION,
            device_id=DEVICE_ID,
            bf16_mode=BF16_MODE,
            cache_dir=CACHE_DIR,
            max_batch_size=MAX_BATCH_SIZE,
        )
        init_time = time.time() - start_time
        logger.info(f"Moondream Engine initialized in {init_time:.2f}s")
        optimal_workers = min(MAX_CONCURRENCY, multiprocessing.cpu_count() + 2)
        executor = ThreadPoolExecutor(
            max_workers=optimal_workers, thread_name_prefix="moondream_worker"
        )
        batch_semaphore = asyncio.Semaphore(optimal_workers)
        for op_type in ["caption", "query", "detect", "point"]:
            request_queues[op_type] = asyncio.Queue()
            queue_locks[op_type] = asyncio.Lock()
            processing_tasks[op_type] = asyncio.create_task(process_queue(op_type))
        logger.info(f"ThreadPool initialized with {optimal_workers} workers")
        logger.info(
            f"Request batch size: {REQUEST_BATCH_SIZE}, Model inference batch size: {MAX_BATCH_SIZE}"
        )
        logger.info(f"Adaptive batching enabled for all operations")
        asyncio.create_task(cleanup_old_batch_jobs())
        logger.info(
            f"Started background task for cleaning up expired batch jobs (max age: {BATCH_JOB_MAX_AGE}s)"
        )

    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    global executor, processing_tasks
    for op_type, task in processing_tasks.items():
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Cancelled {op_type} processing task")
    if executor:
        logger.info("Shutting down thread pool executor...")
        executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Thread pool executor shut down")
    if engine and hasattr(engine, "device") and engine.device.type == "xpu":
        torch.xpu.empty_cache()
        logger.info("XPU cache cleared")
    if torch.xpu.is_available():
        try:
            gc.collect()
            torch.xpu.empty_cache()
            logger.info("Final XPU memory cleanup completed")
        except Exception as e:
            logger.warning(f"Error during final XPU cleanup: {str(e)}")


async def process_image(file: UploadFile) -> Image.Image:
    """Process an uploaded image file into a PIL Image with optimized handling"""
    try:
        MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit
        contents = await file.read(MAX_IMAGE_SIZE)
        if not contents:
            raise ValueError("Empty image file")
        image_bytes = BytesIO(contents)
        image_bytes.seek(0)
        try:
            image = Image.open(image_bytes)
            if image.format not in ["JPEG", "PNG", "WEBP"]:
                logger.warning(
                    f"Unusual image format: {image.format}, converting anyway"
                )
            image = image.convert("RGB")
        except Exception as img_error:
            raise ValueError(f"Invalid image format: {str(img_error)}")
        await file.seek(0)
        return image
    except ValueError as ve:
        logger.error(f"Image validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


async def process_image_batch(
    files: List[UploadFile],
) -> List[Tuple[Image.Image, Optional[Exception]]]:
    """Process multiple uploaded image files into PIL Images in parallel"""
    results = []

    async def _process_one(file, index):
        try:
            image = await process_image(file)
            return index, image, None
        except Exception as e:
            logger.error(f"Error processing image {index}: {str(e)}")
            return index, None, e

    tasks = [_process_one(file, i) for i, file in enumerate(files)]
    raw_results = await asyncio.gather(*tasks, return_exceptions=False)
    sorted_results = sorted(raw_results, key=lambda x: x[0])
    results = [(item[1], item[2]) for item in sorted_results]

    return results


def _run_caption(image, length="normal"):
    result = engine.caption(image, length=length)
    processing_time = result.get("timings", {}).get("total_time", 0.0)
    return {"caption": result.get("caption", ""), "processing_time": processing_time}


def _run_caption_batch(images, lengths):
    """Run caption generation for a batch of images"""
    caption_results = []
    for i, (image, length) in enumerate(zip(images, lengths)):
        try:
            result = engine.caption(image, length)
            processing_time = result.get("timings", {}).get("total_time", 0.0)
            caption_results.append(
                {
                    "caption": result.get("caption", ""),
                    "length": length,
                    "processing_time": processing_time,
                }
            )
        except Exception as e:
            logger.error(f"Error in caption batch processing for image {i}: {str(e)}")
            caption_results.append({"error": str(e), "length": length})
    return caption_results


def _run_query(image, prompt):
    result = engine.query(image, prompt)
    return result


def _run_query_batch(images, prompts):
    """Run query processing for a batch of images"""
    query_results = []
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        try:
            result = engine.query(image, prompt)
            query_results.append(
                {"answer": result["answer"], "timings": result["timings"]}
            )
        except Exception as e:
            logger.error(f"Error in query batch processing for image {i}: {str(e)}")
            query_results.append({"error": str(e)})
    return query_results


def _run_detect(image, object_type):
    result = engine.detect(image, object_type)
    processing_time = result.get("timings", {}).get("total_time", 0.0)
    objects = result.get("objects", [])
    enhanced_objects = []
    for obj in objects:
        if isinstance(obj, dict):
            enhanced_obj = {
                "label": obj.get("label", obj.get("class", object_type)),
                "confidence": obj.get("confidence", obj.get("score", 0.0)),
                "box": obj.get("box", obj.get("bbox", None)),
            }
            enhanced_objects.append(enhanced_obj)
        else:
            enhanced_objects.append(
                {"label": object_type, "confidence": 0.0, "box": None}
            )
    return {"objects": enhanced_objects, "processing_time": processing_time}


def _run_detect_batch(images, object_types):
    """Run object detection for a batch of images"""
    detect_results = []
    for i, (image, obj_type) in enumerate(zip(images, object_types)):
        try:
            result = engine.detect(image, obj_type)
            processing_time = result.get("timings", {}).get("total_time", 0.0)
            objects = result.get("objects", [])
            enhanced_objects = []
            for obj in objects:
                if isinstance(obj, dict):
                    enhanced_obj = {
                        "label": obj.get("label", obj.get("class", obj_type)),
                        "confidence": obj.get("confidence", obj.get("score", 0.0)),
                        "box": obj.get("box", obj.get("bbox", None)),
                    }
                    enhanced_objects.append(enhanced_obj)
                else:
                    enhanced_objects.append(
                        {"label": obj_type, "confidence": 0.0, "box": None}
                    )
            detect_results.append(
                {"objects": enhanced_objects, "processing_time": processing_time}
            )
        except Exception as e:
            logger.error(f"Error in detect batch processing for image {i}: {str(e)}")
            detect_results.append({"error": str(e), "objects": []})
    return detect_results


def _run_point(image, object_type):
    result = engine.point(image, object_type)
    processing_time = result.get("timings", {}).get("total_time", 0.0)
    return {"coordinates": result.get("points", []), "processing_time": processing_time}


def _run_point_batch(images, object_types):
    """Run object pointing for a batch of images"""
    point_results = []
    for i, (image, obj_type) in enumerate(zip(images, object_types)):
        try:
            result = engine.point(image, obj_type)
            processing_time = result.get("timings", {}).get("total_time", 0.0)
            point_results.append(
                {
                    "coordinates": result.get("points", []),
                    "object_type": obj_type,
                    "processing_time": processing_time,
                }
            )
        except Exception as e:
            logger.error(f"Error in point batch processing for image {i}: {str(e)}")
            point_results.append(
                {"error": str(e), "coordinates": None, "object_type": obj_type}
            )
    return point_results


async def run_in_threadpool(func, *args, timeout=None, **kwargs):
    """Run `func` in the thread pool with timeout"""
    loop = asyncio.get_event_loop()
    start_time = time.time()
    if executor:
        future = loop.run_in_executor(executor, lambda: func(*args, **kwargs))
    else:
        future = loop.create_future()
        loop.call_soon(lambda: future.set_result(func(*args, **kwargs)))
    try:
        if timeout:
            result = await asyncio.wait_for(future, timeout=timeout)
        else:
            result = await future
        execution_time = time.time() - start_time
        if execution_time > 5.0:
            func_name = getattr(func, "__name__", str(func))
            logger.warning(f"Slow operation: {func_name} took {execution_time:.2f}s")
        return result
    except asyncio.TimeoutError:
        future.cancel()
        func_name = getattr(func, "__name__", str(func))
        logger.error(f"Operation timed out: {func_name} after {timeout}s")
        raise HTTPException(status_code=504, detail="Processing timed out")
    except Exception as e:
        func_name = getattr(func, "__name__", str(func))
        logger.error(f"Error in threadpool execution of {func_name}: {str(e)}")
        raise


async def process_queue(op_type: str):
    """Background task that processes queued requests in batches"""
    logger.info(f"Started adaptive batching processor for {op_type} operations")
    while True:
        try:
            first_item = await request_queues[op_type].get()
            batch = [first_item]
            request_queues[op_type].task_done()
            batch_timeout = time.time() + BATCH_TIMEOUT
            while (
                len(batch) < REQUEST_BATCH_SIZE
                and time.time() < batch_timeout
                and not request_queues[op_type].empty()
            ):
                try:
                    item = await asyncio.wait_for(request_queues[op_type].get(), 0.01)
                    batch.append(item)
                    request_queues[op_type].task_done()
                except asyncio.TimeoutError:
                    break
            batch_size = len(batch)
            if batch_size > 1:
                logger.info(f"Processing {op_type} batch of size {batch_size}")
            images = [item["image"] for item in batch]
            params = [item["params"] for item in batch]
            futures = [item["future"] for item in batch]
            async with batch_semaphore:
                try:
                    if op_type == "caption":
                        results = await process_caption_batch(images, params)
                    elif op_type == "query":
                        results = await process_query_batch(images, params)
                    elif op_type == "detect":
                        results = await process_detect_batch(images, params)
                    elif op_type == "point":
                        results = await process_point_batch(images, params)
                    else:
                        results = [None] * batch_size
                        logger.error(f"Unknown operation type: {op_type}")
                    for i, future in enumerate(futures):
                        if not future.done():
                            future.set_result(results[i])
                except Exception as e:
                    logger.error(f"Error processing {op_type} batch: {str(e)}")
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                if hasattr(engine, "device") and engine.device.type == "xpu":
                    torch.xpu.empty_cache()
        except asyncio.CancelledError:
            logger.info(f"Adaptive batching processor for {op_type} cancelled")
            break
        except Exception as e:
            logger.error(f"Error in {op_type} queue processor: {str(e)}")
            await asyncio.sleep(0.1)


async def process_caption_batch(images: List[Image.Image], params: List[Dict]):
    """Process a batch of caption requests using batched processing"""
    try:
        lengths = [param.get("length", "normal") for param in params]
        results = await run_in_threadpool(
            _run_caption_batch, images, lengths, timeout=60
        )
        return results
    except Exception as e:
        logger.error(f"Error in batch caption processing: {str(e)}")
        results = []
        for image, param in zip(images, params):
            try:
                result = await run_in_threadpool(
                    _run_caption, image, param.get("length", "normal"), timeout=60
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results


async def process_query_batch(images: List[Image.Image], params: List[Dict]):
    """Process a batch of query requests using batched processing"""
    try:
        prompts = [param.get("prompt", "Describe this image.") for param in params]
        results = await run_in_threadpool(
            _run_query_batch, images, prompts, timeout=120
        )
        return results
    except Exception as e:
        logger.error(f"Error in batch query processing: {str(e)}")
        results = []
        for image, param in zip(images, params):
            try:
                result = await run_in_threadpool(
                    _run_query,
                    image,
                    param.get("prompt", "Describe this image."),
                    timeout=120,
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results


async def process_detect_batch(images: List[Image.Image], params: List[Dict]):
    """Process a batch of detect requests using batched processing"""
    try:
        object_types = [param.get("object_type", "") for param in params]
        results = await run_in_threadpool(
            _run_detect_batch, images, object_types, timeout=120
        )
        return results
    except Exception as e:
        logger.error(f"Error in batch detect processing: {str(e)}")
        results = []
        for image, param in zip(images, params):
            try:
                result = await run_in_threadpool(
                    _run_detect, image, param.get("object_type", ""), timeout=60
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results


async def process_point_batch(images: List[Image.Image], params: List[Dict]):
    """Process a batch of point requests using batched processing"""
    try:
        object_types = [param.get("object_type", "") for param in params]
        results = await run_in_threadpool(
            _run_point_batch, images, object_types, timeout=120
        )
        return results
    except Exception as e:
        logger.error(f"Error in batch point processing: {str(e)}")
        results = []
        for image, param in zip(images, params):
            try:
                result = await run_in_threadpool(
                    _run_point, image, param.get("object_type", ""), timeout=60
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results


async def process_batch_efficiently(operations, images):
    """
    Process a batch of ops, optimized for throughput
    by grouping similar ops and managing XPU resources.
    """
    results = []
    total_time = 0
    operation_groups = {"caption": [], "query": [], "detect": [], "point": []}
    for i, (op, img) in enumerate(zip(operations, images)):
        op_type = op.get("type", "").lower()
        if op_type in operation_groups:
            operation_groups[op_type].append((i, op, img))
    for op_type, op_list in operation_groups.items():
        if not op_list:
            continue
        logger.info(f"Processing {len(op_list)} {op_type} operations")
        async with batch_semaphore:
            batch_start = time.time()
            tasks = []
            for idx, op, img in op_list:
                if op_type == "caption":
                    task = run_in_threadpool(
                        _run_caption, img, op.get("length", "normal")
                    )
                elif op_type == "query":
                    task = run_in_threadpool(
                        _run_query,
                        img,
                        op.get("prompt", "Describe this image."),
                    )
                elif op_type == "detect":
                    task = run_in_threadpool(
                        _run_detect, img, op.get("object_type", "")
                    )
                elif op_type == "point":
                    task = run_in_threadpool(_run_point, img, op.get("object_type", ""))
                tasks.append((idx, task))
            for idx, task in tasks:
                try:
                    op = operations[idx]
                    op_type = op.get("type", "").lower()
                    result = await task
                    if op_type == "caption":
                        results.append(
                            {
                                "index": idx,
                                "type": "caption",
                                "caption": result["caption"],
                            }
                        )
                    elif op_type == "query":
                        results.append(
                            {
                                "index": idx,
                                "type": "query",
                                "prompt": op.get("prompt", ""),
                                "answer": result["answer"],
                            }
                        )
                    elif op_type == "detect":
                        results.append(
                            {
                                "index": idx,
                                "type": "detect",
                                "object_type": op.get("object_type", ""),
                                "objects": result.get("objects", []),
                                "count": len(result.get("objects", [])),
                            }
                        )
                    elif op_type == "point":
                        results.append(
                            {
                                "index": idx,
                                "type": "point",
                                "object_type": op.get("object_type", ""),
                                "points": result.get("points", []),
                                "count": len(result.get("points", [])),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error processing operation {idx}: {str(e)}")
                    results.append({"index": idx, "type": "error", "message": str(e)})
            batch_time = time.time() - batch_start
            total_time += batch_time
            logger.info(f"Processed {op_type} group in {batch_time:.2f}s")
    results.sort(key=lambda x: x["index"])
    if hasattr(engine, "device") and engine.device.type == "xpu":
        torch.xpu.empty_cache()
    return {"results": results, "total_processing_time": total_time}


# =========== Routes ============== #
@app.get("/health")
async def health_check():
    """Check if the API and model are healthy"""
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not initialized"},
        )

    try:
        memory_info = engine.get_memory_stats()
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "device": engine.device_name,
            "xpu_memory": {
                "used_gb": memory_info["used_memory_gb"],
                "total_gb": memory_info["total_memory_gb"],
                "utilization_percent": memory_info["utilization_percent"],
            },
            "system": {"cpu_percent": cpu_usage, "ram_percent": ram_usage},
            "config": {
                "request_batch_size": REQUEST_BATCH_SIZE,
                "model_batch_size": MAX_BATCH_SIZE,
                "max_concurrency": MAX_CONCURRENCY,
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "degraded", "message": str(e)}
        )


@app.post("/caption")
async def caption_image(
    file: UploadFile = File(...),
    length: str = Query("normal", description="Caption length (short/normal)"),
):
    """Generate a caption for an image using adaptive batching"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        start_time = time.time()
        image = await process_image(file)
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request_item = {"image": image, "params": {"length": length}, "future": future}
        await request_queues["caption"].put(request_item)
        try:
            result = await asyncio.wait_for(future, timeout=60)
            process_time = time.time() - start_time
            if isinstance(result, dict) and "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return {"caption": result["caption"], "processing_time": process_time}
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in caption: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
):
    """Answer a question about an image using adaptive batching"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        start_time = time.time()
        image = await process_image(file)
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request_item = {
            "image": image,
            "params": {
                "prompt": prompt,
            },
            "future": future,
        }
        await request_queues["query"].put(request_item)
        try:
            result = await asyncio.wait_for(future, timeout=120)
            process_time = time.time() - start_time
            if isinstance(result, dict) and "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return {"answer": result["answer"], "processing_time": process_time}
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), object_type: str = Form(...)):
    """Detect objects in an image using adaptive batching"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        start_time = time.time()
        image = await process_image(file)
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request_item = {
            "image": image,
            "params": {"object_type": object_type},
            "future": future,
        }
        await request_queues["detect"].put(request_item)
        try:
            result = await asyncio.wait_for(future, timeout=60)
            process_time = time.time() - start_time
            if isinstance(result, dict) and "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            objects = result.get("objects", [])
            enhanced_objects = []
            for obj in objects:
                if isinstance(obj, dict):
                    enhanced_obj = {
                        "label": obj.get("label", obj.get("class", "Unknown")),
                        "confidence": obj.get("confidence", obj.get("score", 0.0)),
                        "box": obj.get("box", obj.get("bbox", None)),
                    }
                    enhanced_objects.append(enhanced_obj)
                else:
                    enhanced_objects.append(
                        {"label": str(obj), "confidence": 0.0, "box": None}
                    )
            return {
                "objects": enhanced_objects,
                "count": len(enhanced_objects),
                "processing_time": process_time,
            }
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/point")
async def point_objects(file: UploadFile = File(...), object_type: str = Form(...)):
    """Point to objects in an image using adaptive batching"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        start_time = time.time()
        image = await process_image(file)
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request_item = {
            "image": image,
            "params": {"object_type": object_type},
            "future": future,
        }
        await request_queues["point"].put(request_item)
        try:
            result = await asyncio.wait_for(future, timeout=60)
            process_time = time.time() - start_time
            if isinstance(result, dict) and "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return {
                "points": result.get("points", []),
                "count": len(result.get("points", [])),
                "processing_time": process_time,
            }
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in point: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_process(
    files: List[UploadFile] = File(...),
    operations: str = Form(..., description="JSON string of operations to perform"),
):
    """Process multiple operations in a batch with adaptive batching (synchronous)"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    # Check batch size
    if len(files) > REQUEST_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(files)} exceeds maximum allowed size of {REQUEST_BATCH_SIZE}",
        )
    try:
        try:
            ops = json.loads(operations)
            if not isinstance(ops, list) or len(ops) != len(files):
                raise ValueError(
                    "Operations must be a list matching the number of files"
                )
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in operations")
        start_time = time.time()
        image_results = await process_image_batch(files)
        images = []
        valid_ops = []
        errors = []
        for i, (image, error) in enumerate(image_results):
            if error:
                errors.append(
                    {
                        "index": i,
                        "type": "error",
                        "message": f"Image processing error: {str(error)}",
                    }
                )
            else:
                images.append(image)
                valid_ops.append(ops[i])
        if not images:
            return {
                "results": errors,
                "total_processing_time": 0,
                "status": "failed",
                "message": "All images failed to process",
            }
        futures = []
        op_indices = []
        for i, (op, img) in enumerate(zip(valid_ops, images)):
            op_type = op.get("type", "").lower()
            if op_type not in ["caption", "query", "detect", "point"]:
                errors.append(
                    {
                        "index": i,
                        "type": "error",
                        "message": f"Unknown operation type: {op_type}",
                    }
                )
                continue
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            params = {}
            if op_type == "caption":
                params = {"length": op.get("length", "normal")}
            elif op_type == "query":
                params = {
                    "prompt": op.get("prompt", ""),
                }
            elif op_type in ["detect", "point"]:
                params = {"object_type": op.get("object_type", "")}
            request_item = {"image": img, "params": params, "future": future}
            await request_queues[op_type].put(request_item)
            futures.append(future)
            op_indices.append(i)
        try:
            batch_results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=180,  # 3 minutes timeout for batch
            )
            results = []
            for i, (idx, result) in enumerate(zip(op_indices, batch_results)):
                op = valid_ops[i]
                op_type = op.get("type", "").lower()
                if isinstance(result, Exception):
                    results.append(
                        {"index": idx, "type": "error", "message": str(result)}
                    )
                    continue
                if op_type == "caption":
                    results.append(
                        {
                            "index": idx,
                            "type": "caption",
                            "caption": result.get("caption", ""),
                        }
                    )
                elif op_type == "query":
                    results.append(
                        {
                            "index": idx,
                            "type": "query",
                            "prompt": op.get("prompt", ""),
                            "answer": result.get("answer", ""),
                        }
                    )
                elif op_type == "detect":
                    results.append(
                        {
                            "index": idx,
                            "type": "detect",
                            "object_type": op.get("object_type", ""),
                            "objects": result.get("objects", []),
                            "count": len(result.get("objects", [])),
                        }
                    )
                elif op_type == "point":
                    results.append(
                        {
                            "index": idx,
                            "type": "point",
                            "object_type": op.get("object_type", ""),
                            "points": result.get("points", []),
                            "count": len(result.get("points", [])),
                        }
                    )
            results.extend(errors)
            results.sort(key=lambda x: x["index"])
            process_time = time.time() - start_time
            return {
                "results": results,
                "total_processing_time": process_time,
                "status": "success",
                "total_time_with_loading": process_time,
            }

        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Batch processing timed out")

    except Exception as e:
        logger.error(f"Error in batch_process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/async")
async def batch_process_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    operations: str = Form(..., description="JSON string of operations to perform"),
):
    """
    Process multiple ops in a batch asynchronously with adaptive batching.
    Returns a job ID that can be used to check the status and retrieve results.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    if len(files) > REQUEST_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(files)} exceeds maximum allowed size of {REQUEST_BATCH_SIZE}",
        )
    try:
        try:
            ops = json.loads(operations)
            if not isinstance(ops, list) or len(ops) != len(files):
                raise ValueError(
                    "Operations must be a list matching the number of files"
                )
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in operations")
        temp_file_paths = []
        for i, file in enumerate(files):
            os.makedirs("./temp_uploads", exist_ok=True)
            temp_path = f"./temp_uploads/{uuid.uuid4()}_{file.filename}"
            contents = await file.read()
            with open(temp_path, "wb") as f:
                f.write(contents)
            temp_file_paths.append(temp_path)
            await file.seek(0)
        job_id = f"batch_{uuid.uuid4().hex}"
        current_time = time.time()
        new_job = BatchJob(
            job_id=job_id,
            status="initializing",
            created_at=current_time,
            results=None,
            error=None,
            completed_at=None,
        )
        batch_jobs[job_id] = new_job
        if job_id in batch_jobs:
            logger.info(f"Created batch async job: {job_id} with {len(files)} files")
            logger.info(f"Current batch jobs after creation: {list(batch_jobs.keys())}")
        else:
            logger.error(f"Failed to store job {job_id} in batch_jobs dictionary")

        async def process_batch_job():
            try:
                batch_jobs[job_id].status = "processing"
                logger.info(f"Started processing batch job: {job_id}")
                images = []
                errors = []
                logger.info(f"Loading {len(temp_file_paths)} images for job {job_id}")
                for i, path in enumerate(temp_file_paths):
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                        errors.append(None)
                    except Exception as e:
                        logger.error(
                            f"Error processing image {i} in job {job_id}: {str(e)}"
                        )
                        images.append(None)
                        errors.append(e)
                results = [None] * len(ops)
                for i, error in enumerate(errors):
                    if error:
                        results[i] = {
                            "index": i,
                            "type": "error",
                            "message": f"Image processing error: {str(error)}",
                        }
                logger.info(f"Processing {len(ops)} operations for job {job_id}")
                op_groups = {"caption": [], "query": [], "detect": [], "point": []}
                op_indices = {"caption": [], "query": [], "detect": [], "point": []}
                for i, (op, img) in enumerate(zip(ops, images)):
                    if img is None:
                        continue  # Skip failed images
                    op_type = op.get("type", "").lower()
                    if op_type not in ["caption", "query", "detect", "point"]:
                        continue
                    if op_type == "caption":
                        op_groups["caption"].append((img, op.get("length", "normal")))
                        op_indices["caption"].append(i)
                    elif op_type == "query":
                        op_groups["query"].append((img, op.get("prompt", "")))
                        op_indices["query"].append(i)
                    elif op_type == "detect":
                        op_groups["detect"].append((img, op.get("object_type", "")))
                        op_indices["detect"].append(i)
                    elif op_type == "point":
                        op_groups["point"].append((img, op.get("object_type", "")))
                        op_indices["point"].append(i)
                for op_type, op_list in op_groups.items():
                    if not op_list:
                        continue
                    logger.info(
                        f"Processing {len(op_list)} {op_type} operations for job {job_id}"
                    )
                    for i in range(0, len(op_list), engine.max_batch_size):
                        batch = op_list[i : i + engine.max_batch_size]
                        batch_indices = op_indices[op_type][
                            i : i + engine.max_batch_size
                        ]

                        try:
                            if op_type == "caption":
                                batch_images = [item[0] for item in batch]
                                batch_lengths = [item[1] for item in batch]
                                batch_results = await run_in_threadpool(
                                    _run_caption_batch, batch_images, batch_lengths
                                )
                                for idx, result in zip(batch_indices, batch_results):
                                    results[idx] = {
                                        "index": idx,
                                        "type": "caption",
                                        "caption": result.get("caption", ""),
                                    }

                            elif op_type == "query":
                                batch_images = [item[0] for item in batch]
                                batch_prompts = [item[1] for item in batch]
                                batch_results = await run_in_threadpool(
                                    _run_query_batch, batch_images, batch_prompts
                                )
                                for idx, result, (_, prompt) in zip(
                                    batch_indices, batch_results, batch
                                ):
                                    results[idx] = {
                                        "index": idx,
                                        "type": "query",
                                        "prompt": prompt,
                                        "answer": result.get("answer", ""),
                                    }

                            elif op_type == "detect":
                                batch_images = [item[0] for item in batch]
                                batch_object_types = [item[1] for item in batch]
                                batch_results = await run_in_threadpool(
                                    _run_detect_batch, batch_images, batch_object_types
                                )
                                for idx, result, (_, object_type) in zip(
                                    batch_indices, batch_results, batch
                                ):
                                    results[idx] = {
                                        "index": idx,
                                        "type": "detect",
                                        "object_type": object_type,
                                        "objects": result.get("objects", []),
                                        "count": len(result.get("objects", [])),
                                    }

                            elif op_type == "point":
                                batch_images = [item[0] for item in batch]
                                batch_object_types = [item[1] for item in batch]
                                batch_results = await run_in_threadpool(
                                    _run_point_batch, batch_images, batch_object_types
                                )
                                for idx, result, (_, object_type) in zip(
                                    batch_indices, batch_results, batch
                                ):
                                    results[idx] = {
                                        "index": idx,
                                        "type": "point",
                                        "object_type": object_type,
                                        "points": result.get("points", []),
                                        "count": len(result.get("points", [])),
                                    }
                        except Exception as e:
                            logger.error(
                                f"Error processing {op_type} batch in job {job_id}: {str(e)}"
                            )
                            for idx in batch_indices:
                                results[idx] = {
                                    "index": idx,
                                    "type": "error",
                                    "message": f"Error processing {op_type}: {str(e)}",
                                }
                        if hasattr(engine, "device") and engine.device.type == "xpu":
                            torch.xpu.empty_cache()
                results = [r for r in results if r is not None]
                if job_id in batch_jobs:
                    # Debug: Print all job IDs before updating
                    logger.debug(
                        f"Before updating job {job_id}, current batch jobs: {list(batch_jobs.keys())}"
                    )
                    job = batch_jobs[job_id]
                    job.status = "completed"
                    job.results = results
                    job.completed_at = time.time()
                    elapsed = job.completed_at - job.created_at
                    expiry_time = job.completed_at + COMPLETED_JOB_RETENTION
                    logger.info(
                        f"Completed processing job {job_id} with {len(results)} results in {elapsed:.2f}s"
                    )
                    logger.info(
                        f"Job {job_id} will be retained until {time.ctime(expiry_time)} ({COMPLETED_JOB_RETENTION}s from now)"
                    )
                    job.results = results
                    batch_jobs[job_id] = job
                    logger.info(
                        f"After updating job {job_id}, current batch jobs: {list(batch_jobs.keys())}"
                    )
                    logger.info(
                        f"Job {job_id} status is now: {batch_jobs[job_id].status}"
                    )
                else:
                    logger.error(
                        f"Job {job_id} was removed before completion could be recorded"
                    )
                for path in temp_file_paths:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {path}: {str(e)}")
            except Exception as e:
                logger.error(f"Error in async batch job {job_id}: {str(e)}")
                if job_id in batch_jobs:
                    batch_jobs[job_id].status = "failed"
                    batch_jobs[job_id].error = str(e)
                    batch_jobs[job_id].completed_at = time.time()
                for path in temp_file_paths:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Failed to remove temp file {path}: {str(cleanup_error)}"
                        )

        async def process_and_verify():
            if job_id in batch_jobs:
                logger.info(f"Starting background task for job {job_id}")
                await process_batch_job()
                if job_id in batch_jobs:
                    logger.info(
                        f"Job {job_id} still exists after processing (status: {batch_jobs[job_id].status})"
                    )
                else:
                    logger.error(f"Job {job_id} was removed during or after processing")
            else:
                logger.error(f"Job {job_id} not found before starting background task")

        background_tasks.add_task(process_and_verify)
        logger.info(f"Added batch job {job_id} to background tasks")
        return {"job_id": job_id, "status": "initializing"}
    except Exception as e:
        logger.error(f"Error in batch_process_async: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/status/{job_id}")
async def get_batch_status(job_id: str):
    """
    Get the status of an aasync batch processing job.
    """
    logger.info(f"Current batch jobs: {list(batch_jobs.keys())}")
    if job_id not in batch_jobs:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    job = batch_jobs[job_id]
    logger.info(f"Retrieved job {job_id} with status: {job.status}")
    current_time = time.time()
    elapsed = current_time - job.created_at
    if job.status in ["completed", "failed"] and hasattr(job, "completed_at"):
        retention_limit = job.completed_at + COMPLETED_JOB_RETENTION
        time_until_cleanup = max(0, retention_limit - current_time)
    else:
        retention_limit = job.created_at + BATCH_JOB_MAX_AGE
        time_until_cleanup = max(0, retention_limit - current_time)
    response = {
        "job_id": job.job_id,
        "status": job.status,
        "elapsed_seconds": elapsed,
        "created_at": time.ctime(job.created_at),
        "retention_seconds_remaining": time_until_cleanup,
        "will_be_removed_at": time.ctime(retention_limit),
    }
    if hasattr(job, "completed_at") and job.completed_at:
        response["completed_at"] = time.ctime(job.completed_at)
        response["processing_seconds"] = job.completed_at - job.created_at
    if job.status == "initializing":
        response["status_description"] = "Job is being initialized"
    elif job.status == "processing":
        response["status_description"] = "Job is currently being processed"
    elif job.status == "completed":
        response["status_description"] = "Job has completed successfully"
    elif job.status == "failed":
        response["status_description"] = "Job processing failed"
    if job.results:
        response["results"] = job.results
    if job.error:
        response["error"] = job.error
    return response


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=WORKERS,  # Number of server workers (1 for 1 GPU - no multiple GPU load balancing)
        limit_concurrency=BACKLOG,  # Limit concurrent connections
        timeout_keep_alive=KEEP_ALIVE,  # Keep-alive timeout
        backlog=BACKLOG,  # Connection queue size
        timeout_graceful_shutdown=GRACEFUL_TIMEOUT,  # Graceful shutdown timeout
        log_level="info",  # Log level
        access_log=True,  # Enable access logging
        proxy_headers=True,
    )
