#!/usr/bin/env python3
"""
Moondream API Demo Client

This script shows how to use all the Moondream API endpoints with  examples.

Usage:
    python demo_client.py [--url http://localhost:8000]

Requirements:
    pip install requests aiohttp
"""

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

import aiohttp
import requests

# ===== SERVER CONFIG =====
DEFAULT_URL = "http://localhost:8000"

# Dictionary of image URLs to test
IMAGE_URLS = {
    "city.jpg": "https://images.pexels.com/photos/374870/pexels-photo-374870.jpeg",
    "people.jpg": "https://images.pexels.com/photos/1157557/pexels-photo-1157557.jpeg",
    "dog.jpg": "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
    "cat.jpg": "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
}

# Test prompts to use with images
TEST_PROMPTS = [
    "What can you see in this image?",
    "Is there a person in this image?",
    "Describe the main subject of this image.",
    "What colors are prominent in this image?",
]


def download_images(urls_dict, output_dir=None):
    """Download images from URLs and save them to a directory.
    Args:
        urls_dict: Dictionary of {filename: url}
        output_dir: Directory to save images (if None, uses temp directory)
    Returns:
        Dictionary of {filename: local_path}
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="moondream_images_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    print(f"\n===== DOWNLOADING IMAGES =====")
    print(f"Output directory: {output_dir}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
    }

    image_paths = {}
    for filename, url in urls_dict.items():
        output_path = os.path.join(output_dir, filename)
        try:
            print(f"Downloading {url} to {output_path}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            image_paths[filename] = output_path
            print(f"✓ Successfully downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    return image_paths


def use_local_images(input_dir):
    """Use images from a local directory instead of downloading.
    Args:
        input_dir: Directory containing images
    Returns:
        Dictionary of {filename: local_path}
    """
    print(f"\n===== USING LOCAL IMAGES =====")
    print(f"Input directory: {input_dir}")
    image_paths = {}
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        for image_path in Path(input_dir).glob(f"*{ext}"):
            filename = image_path.name
            image_paths[filename] = str(image_path)
            print(f"Found local image: {filename}")
    return image_paths


# ===== SYNCHRONOUS API EXAMPLES =====
def caption_image(url, image_path, length="normal"):
    """Generate a caption for an image.
    Args:
        url: API base URL
        image_path: Path to image file
        length: Caption length (short, normal, long)
    Returns:
        Caption text
    """
    print(f"\n===== CAPTION IMAGE (SYNC) =====")
    print(f"Image: {image_path}, Length: {length}")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"length": length}
            response = requests.post(f"{url}/caption", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            print(f"Caption: {result['caption']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            return result
    except Exception as e:
        print(f"Error captioning image: {e}")
        return None


def query_image(url, image_path, prompt):
    """Ask a question about an image.
    Args:
        url: API base URL
        image_path: Path to image file
        prompt: Question to ask about the image
    Returns:
        Answer text
    """
    print(f"\n===== QUERY IMAGE (SYNC) =====")
    print(f"Image: {image_path}, Prompt: {prompt}")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"prompt": prompt}
            response = requests.post(f"{url}/query", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            print(f"Answer: {result['answer']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            return result
    except Exception as e:
        print(f"Error querying image: {e}")
        return None


def detect_objects(url, image_path, object_type):
    """Detect objects of a specific type in an image.
    Args:
        url: API base URL
        image_path: Path to image file
        object_type: Type of object to detect (e.g., "person", "car")
    Returns:
        List of detected objects with bounding boxes
    """
    print(f"\n===== DETECT OBJECTS (SYNC) =====")
    print(f"Image: {image_path}, Object type: {object_type}")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"object_type": object_type}
            response = requests.post(f"{url}/detect", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            objects = result.get("objects", [])
            print(f"Found {len(objects)} objects")
            for i, obj in enumerate(objects):
                if isinstance(obj, dict):
                    label = obj.get("label", obj.get("class", "Unknown"))
                    confidence = obj.get("confidence", obj.get("score", 0.0))
                    box = obj.get("box", obj.get("bbox", "Unknown"))
                    print(f"  - {label} (confidence: {confidence:.2f})")
                    print(f"    Box: {box}")
                elif isinstance(obj, str):
                    print(f"  - {obj}")
                else:
                    print(f"  - Object {i}: {obj}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            return result
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return None


def batch_process(url, image_paths, operations):
    """Process multiple images with different operations in a single batch.
    Args:
        url: API base URL
        image_paths: List of paths to image files
        operations: List of operations to perform on each image
    Returns:
        Batch processing results
    """
    if not image_paths:
        print("No images available for batch processing")
        return None
    if len(operations) != len(image_paths):
        print(
            f"Warning: Number of operations ({len(operations)}) doesn't match number of images ({len(image_paths)})"
        )
        if len(operations) == 1 and len(image_paths) > 1:
            operations = operations * len(image_paths)
        elif len(operations) < len(image_paths):
            operations = (operations * (len(image_paths) // len(operations) + 1))[
                : len(image_paths)
            ]

    print(f"\n===== BATCH PROCESSING (SYNC) =====")
    print(f"Number of images: {len(image_paths)}")

    try:
        file_handles = []
        files = []
        try:
            for i, path in enumerate(image_paths):
                file_handle = open(path, "rb")
                file_handles.append(file_handle)
                files.append(
                    ("files", (os.path.basename(path), file_handle, "image/jpeg"))
                )
            data = {"operations": json.dumps(operations)}
            response = requests.post(f"{url}/batch", files=files, data=data)
            response.raise_for_status()
            result = response.json()

            print(
                f"Batch processing completed in {result['total_processing_time']:.2f} seconds"
            )
            print(f"Results: {json.dumps(result['results'], indent=2)}")
            return result
        finally:
            for handle in file_handles:
                try:
                    handle.close()
                except:
                    pass
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return None


# ===== ASYNCHRONOUS API EXAMPLES =====
async def caption_image_async(url, image_path, length="normal"):
    """Generate a caption for an image asynchronously.
    Args:
        url: API base URL
        image_path: Path to image file
        length: Caption length (short, normal, long)
    Returns:
        Caption text
    """
    print(f"\n===== CAPTION IMAGE (ASYNC) =====")
    print(f"Image: {image_path}, Length: {length}")
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                open(image_path, "rb"),
                filename=os.path.basename(image_path),
                content_type="image/jpeg",
            )
            data.add_field("length", length)
            async with session.post(f"{url}/caption", data=data) as response:
                response.raise_for_status()
                result = await response.json()
                print(f"Caption: {result['caption']}")
                print(f"Processing time: {result['processing_time']:.2f} seconds")
                return result
    except Exception as e:
        print(f"Error in async caption: {e}")
        return None


async def batch_process_async(url, image_paths, operations):
    """Process multiple images asynchronously with different operations.
    Args:
        url: API base URL
        image_paths: List of paths to image files
        operations: List of operations to perform on each image
    Returns:
        Job ID for the batch processing task
    """
    if not image_paths:
        print("No images available for async batch processing")
        return None
    if len(operations) != len(image_paths):
        print(
            f"Warning: Number of operations ({len(operations)}) doesn't match number of images ({len(image_paths)})"
        )
        if len(operations) == 1 and len(image_paths) > 1:
            operations = operations * len(image_paths)
        elif len(operations) < len(image_paths):
            operations = (operations * (len(image_paths) // len(operations) + 1))[
                : len(image_paths)
            ]

    print(f"\n===== BATCH PROCESSING (ASYNC) =====")
    print(f"Number of images: {len(image_paths)}")

    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            for i, path in enumerate(image_paths):
                try:
                    with open(path, "rb") as f:
                        file_content = f.read()
                    data.add_field(
                        name="files",
                        value=file_content,
                        filename=os.path.basename(path),
                        content_type="image/jpeg",
                    )
                except Exception as e:
                    print(f"Error reading file {path}: {e}")
                    return None
            data.add_field("operations", json.dumps(operations))
            try:
                async with session.post(f"{url}/batch/async", data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Error: {response.status} - {error_text}")
                        return None
                    result = await response.json()
                    print(f"Job ID: {result['job_id']}")
                    print(f"Status: {result['status']}")
                    job_id = result["job_id"]
                    await wait_for_batch_completion(url, job_id)
                    return result
            except aiohttp.ClientResponseError as e:
                print(f"HTTP error in batch async request: {e.status} - {e.message}")
                return None
            except Exception as e:
                print(f"Error in batch async request: {e}")
                return None
    except Exception as e:
        print(f"Error in async batch processing: {e}")
        return None


async def wait_for_batch_completion(url, job_id, max_retries=30, retry_delay=1.0):
    """Wait for an asynchronous batch job to complete.
    Args:
        url: API base URL
        job_id: Batch job ID to check
        max_retries: Maximum number of status check attempts
        retry_delay: Delay between status checks in seconds
    Returns:
        Final job status and results
    """
    print(f"Waiting for batch job completion: {job_id}")
    await asyncio.sleep(2.0)
    consecutive_not_found = 0
    max_consecutive_not_found = 5
    for retry in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/batch/status/{job_id}") as response:
                    if response.status == 404:
                        consecutive_not_found += 1
                        print(
                            f"Job {job_id} not found (attempt {retry+1}/{max_retries}, consecutive: {consecutive_not_found}/{max_consecutive_not_found})"
                        )
                        if consecutive_not_found >= max_consecutive_not_found:
                            print(
                                f"Job {job_id} not found after {consecutive_not_found} consecutive attempts. It may have been cleaned up or never existed."
                            )
                            return {
                                "status": "not_found",
                                "job_id": job_id,
                                "error": "Job not found after multiple attempts",
                            }
                        current_delay = retry_delay * (1 + 0.5 * retry)
                        print(f"Waiting {current_delay:.1f} seconds...")
                        await asyncio.sleep(current_delay)
                        continue
                    response.raise_for_status()
                    result = await response.json()
                    consecutive_not_found = 0
                    if "retention_seconds_remaining" in result:
                        retention_time = result["retention_seconds_remaining"]
                        print(
                            f"Job will be retained for {retention_time:.1f} more seconds"
                        )
                    if result["status"] == "completed":
                        completion_time = result.get(
                            "processing_seconds", result.get("elapsed_seconds", 0)
                        )
                        print(f"Job completed in {completion_time:.2f} seconds")
                        if "results" in result:
                            print(f"Results: {json.dumps(result['results'], indent=2)}")
                        return result
                    if result["status"] == "failed":
                        print(f"Job failed: {result.get('error', 'Unknown error')}")
                        return result
                    status_desc = result.get("status_description", result["status"])
                    elapsed = result.get("elapsed_seconds", 0)
                    print(
                        f"Job status: {result['status']} - {status_desc} for {elapsed:.1f}s (attempt {retry+1}/{max_retries})"
                    )
                    current_delay = min(
                        retry_delay * (1 + 0.2 * retry), 55.0
                    )  # Cap at 55 seconds
                    print(f"Waiting {current_delay:.1f} seconds...")
                    await asyncio.sleep(current_delay)
        except aiohttp.ClientResponseError as e:
            print(f"HTTP error checking batch status: {e.status} - {e.message}")
            await asyncio.sleep(retry_delay * (1 + 0.5 * retry))
        except Exception as e:
            print(f"Error checking batch status: {e}")
            await asyncio.sleep(retry_delay * (1 + 0.5 * retry))
    print(f"Timed out waiting for job {job_id} to complete")
    return None


def smart_object_detection(url, image_path):
    """First caption the image to identify what's in it, then detect relevant objects."""
    print(f"\n===== SMART OBJECT DETECTION =====")
    print(f"Image: {image_path}")
    caption_result = caption_image(url, image_path)
    if not caption_result or "caption" not in caption_result:
        print("Could not get caption for image, falling back to standard detection")
        detect_types = ["person", "animal", "vehicle", "building"]
        for obj_type in detect_types:
            detect_objects(url, image_path, obj_type)
        return
    caption = caption_result["caption"]
    print(f"\nImage caption: {caption}")
    print("Analyzing caption to identify potential objects...")
    # example mappings
    object_mapping = {
        "person": ["person", "man", "woman", "child", "boy", "girl", "people", "human"],
        "animal": [
            "animal",
            "cat",
            "dog",
            "bird",
            "pet",
            "wildlife",
            "horse",
            "cow",
        ],
        "vehicle": [
            "car",
            "vehicle",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "train",
            "boat",
        ],
        "building": [
            "building",
            "house",
            "structure",
            "architecture",
            "home",
            "tower",
            "skyscraper",
        ],
    }
    caption_lower = caption.lower()
    detected_types = []

    for obj_type, keywords in object_mapping.items():
        for keyword in keywords:
            if keyword in caption_lower:
                detected_types.append(obj_type)
                print(
                    f"Caption suggests searching for: {obj_type} (keyword: {keyword})"
                )
                break
    if not detected_types:
        print("No specific objects identified in caption, trying all object types")
        detected_types = list(object_mapping.keys())
    print("\nPerforming targeted object detection:")
    for obj_type in detected_types:
        detect_objects(url, image_path, obj_type)


def run_sync_examples_on_all_images(url, image_paths):
    """Run all synchronous API examples on all images."""
    if not image_paths:
        print("No images available for testing")
        return

    for image_name, image_path in image_paths.items():
        print(f"\n\n==================================================")
        print(f"TESTING IMAGE: {image_name}")
        print(f"======================================================")
        caption_image(url, image_path)
        for prompt in TEST_PROMPTS:
            query_image(url, image_path, prompt)
        smart_object_detection(url, image_path)
    all_paths = list(image_paths.values())
    if all_paths:
        batch_paths = all_paths[: min(2, len(all_paths))]
        operations = []
        for i in range(len(batch_paths)):
            if i % 2 == 0:
                operations.append({"type": "caption", "length": "short"})
            else:
                operations.append(
                    {
                        "type": "query",
                        "prompt": "Summarize what's in this image in one sentence.",
                    }
                )
        batch_process(url, batch_paths, operations)


async def run_async_examples_on_sample(url, image_paths):
    """Run all asynchronous API examples on a sample image."""
    if not image_paths:
        print("No images available for async testing")
        return
    sample_image = list(image_paths.values())[0]
    await caption_image_async(url, sample_image)  # block

    operations = [
        {"type": "caption", "length": "short"},
        {"type": "query", "prompt": "What time of day is it?"},
    ]
    sample_images = list(image_paths.values())[: min(2, len(image_paths))]
    await batch_process_async(url, sample_images, operations)  # block


async def main():
    """Main function to run the demo client."""
    parser = argparse.ArgumentParser(description="Moondream API Demo Client")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL for the API")
    parser.add_argument(
        "--sync-only", action="store_true", help="Run only synchronous examples"
    )
    parser.add_argument(
        "--async-only", action="store_true", help="Run only asynchronous examples"
    )
    parser.add_argument("--download-dir", help="Directory to save downloaded images")
    parser.add_argument(
        "--image-dir",
        help="Directory containing local images to use instead of downloading",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local images from the current directory",
    )
    args = parser.parse_args()

    image_paths = {}
    if args.use_local or args.image_dir:
        image_dir = args.image_dir if args.image_dir else "."
        image_paths = use_local_images(image_dir)
    else:
        try:
            image_paths = download_images(IMAGE_URLS, args.download_dir)
        except Exception as e:
            print(f"Error downloading images: {e}")
        if not image_paths:
            print(
                "Failed to download images. Looking for local images in current directory as fallback..."
            )
            image_paths = use_local_images(".")
    if not image_paths:
        print("Error: No images available for testing")
        return
    print(f"Moondream API Demo Client - Server: {args.url}")
    print(f"Using {len(image_paths)} images for testing")
    if not args.async_only:
        run_sync_examples_on_all_images(args.url, image_paths)
    if not args.sync_only:
        await run_async_examples_on_sample(args.url, image_paths)
    print("\n===== DEMO COMPLETED =====")


if __name__ == "__main__":
    asyncio.run(main())
