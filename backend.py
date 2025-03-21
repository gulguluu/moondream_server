import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import intel_extension_for_pytorch as ipex
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForCausalLM

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


class MoondreamEngine:
    """
    Inference engine for Moondream 2B on Intel Max Series GPUs.
    """

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: Optional[str] = None,  # Using Main version
        device_id: int = 0,  # XPU number
        bf16_mode: bool = True,
        optimization_level: str = "O1",  # default set : conv+bn folding, weights prepack, dropout removal
        max_batch_size: int = 16,  # if seeing OOM, reduce this by using MAX_BATCH_SIZE env variable in DOCKERFILE or setting it here.
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the optimized Moondream inference engine.

        Args:
            model_id: HuggingFace model ID
            revision: Model revision to use (None = main branch)
            device_id: Intel GPU device ID (default: 0)
            bf16_mode: Whether to use BF16 precision for inference (default: True)
            optimization_level: IPEX optimization level (O0, O1, O2) (default: O1)
            max_batch_size: Maximum batch size for inference optimization (default: 16)
            cache_dir: Directory to cache models (default: None)
        """
        self.start_time = time.time()
        logger.info(
            f"Initializing Moondream Engine (model: {model_id}, revision: {revision or 'main'})"
        )
        self.max_batch_size = max_batch_size
        logger.info(f"Using max batch size: {self.max_batch_size}")
        self._setup_device(device_id)
        self.model = self._load_model(
            model_id=model_id,
            revision=revision,
            bf16_mode=bf16_mode,
            optimization_level=optimization_level,
            cache_dir=cache_dir,
        )
        self.initialized = True
        init_time = time.time() - self.start_time
        logger.success(f"Moondream Engine initialized in {init_time:.2f}s")
        self._warmup()

    def _setup_device(self, device_id: int):
        """Set up Intel XPU device with proper environment configuration"""
        os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{device_id}"
        if torch.xpu.is_available():
            self.device = torch.device("xpu")
            self.device_name = torch.xpu.get_device_name()
            logger.info(f"Using Intel XPU: {self.device_name}")
            free_mem, total_mem = torch.xpu.mem_get_info()
            self.total_memory_gb = total_mem / (1024**3)
            self.free_memory_gb = free_mem / (1024**3)
            logger.info(
                f"GPU Memory: {self.free_memory_gb:.2f}GB free / {self.total_memory_gb:.2f}GB total"
            )
        else:
            self.device = torch.device("cpu")
            logger.warning("Intel XPU not available, falling back to CPU")
            self.device_name = "CPU"
            self.total_memory_gb = 0
            self.free_memory_gb = 0

    def _load_model(
        self,
        model_id: str,
        revision: Optional[str],
        bf16_mode: bool,
        optimization_level: str,
        cache_dir: Optional[str],
    ):
        """Load and optimize the model with advanced Intel optimizations"""
        ipex_config = {"level": optimization_level}
        logger.info(f"Using IPEX optimization level: {optimization_level}")
        if bf16_mode and self.device.type == "xpu" and torch.xpu.is_bf16_supported():
            logger.info("Using BF16 precision for inference")
            self.dtype = torch.bfloat16
        else:
            logger.info("Using FP32 precision for inference")
            self.dtype = torch.float32
        logger.info(f"Loading model: {model_id}")
        load_start = time.time()
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "device_map": {"": self.device},
            "cache_dir": cache_dir,
        }
        if revision is not None:
            model_kwargs["revision"] = revision

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except Exception as e:
            if revision is not None:
                logger.warning(f"Failed to load with revision {revision}: {str(e)}")
                logger.info("Attempting to load from main branch...")
                model_kwargs.pop("revision", None)
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            else:
                raise  # model loading failed
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f}s")
        if self.device.type == "xpu":
            optimize_start = time.time()
            logger.info("Applying IPEX optimizations...")
            model = ipex.optimize(model, dtype=self.dtype, inplace=True, **ipex_config)
            optimize_time = time.time() - optimize_start
            logger.info(f"IPEX optimizations applied in {optimize_time:.2f}s")
        return model

    def _warmup(self, num_warmup_rounds: int = 4):
        """Perform warmup inference to prime JIT compilation and caches"""
        logger.info(f"Running {num_warmup_rounds} warmup iterations...")
        dummy_image = Image.new("RGB", (224, 224), color="white")
        warmup_start = time.time()
        for i in range(num_warmup_rounds):
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, dtype=self.dtype
            ):
                encoded = self.model.encode_image(dummy_image)
                _ = self.model.query(encoded, "Describe this image.")
                if self.device.type == "xpu":
                    torch.xpu.empty_cache()
        warmup_time = time.time() - warmup_start
        logger.info(f"Warmup completed in {warmup_time:.2f}s")

    def _open_image(self, image: Union[Image.Image, str, bytes, Path]) -> Image.Image:
        """
        Helper method to open an image from various formats.

        Args:
            image: PIL Image, file path, bytes, or Path object

        Returns:
            PIL Image object

        Raises:
            TypeError: If the image format is not supported
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB")
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def encode_image(self, image: Union[Image.Image, str, bytes, Path]) -> torch.Tensor:
        """
        Encode image for querying.

        Args:
            image: PIL Image, file path, or bytes containing the image

        Returns:
            Tensor containing the encoded image
        """
        image_pil = self._open_image(image)
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype
        ):
            encoded = self.model.encode_image(image_pil)
        return encoded

    def batch_encode_images(
        self, images: List[Union[Image.Image, str, bytes, Path]]
    ) -> List[torch.Tensor]:
        """
        Encode multiple images in a single batch for more efficient processing.

        Args:
            images: List of images to encode (PIL Images, file paths, or bytes)

        Returns:
            List of encoded image tensors
        """
        if not images:
            return []
        pil_images = [self._open_image(img) for img in images]
        batch_size = min(self.max_batch_size, len(pil_images))
        logger.info(f"Processing {len(pil_images)} images with batch size {batch_size}")
        all_encoded = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, dtype=self.dtype
            ):
                batch_encoded = [self.model.encode_image(img) for img in batch]
                all_encoded.extend(batch_encoded)
            if self.device.type == "xpu" and i + batch_size < len(pil_images):
                torch.xpu.empty_cache()
        return all_encoded

    def query(
        self,
        image_or_encoded: Union[torch.Tensor, Image.Image, str, bytes, Path],
        prompt: str = None,
    ) -> Dict[str, Union[str, float]]:
        """
        Query the model with an image and text prompt.

        Args:
            image_or_encoded: Image (various formats) or pre-encoded image tensor
            prompt: Text prompt to process with the image (default: "Describe this image.")

        Returns:
            Dictionary with answer and timing information
        """
        if prompt is None:
            prompt = "Describe this image."
        if not isinstance(prompt, str):
            prompt = str(prompt)
        start_time = time.time()
        timings = {}
        if isinstance(image_or_encoded, torch.Tensor):
            encoded = image_or_encoded
            timings["encode_time"] = 0.0
        else:
            encode_start = time.time()
            encoded = self.encode_image(image_or_encoded)
            encode_time = time.time() - encode_start
            timings["encode_time"] = encode_time

        query_start = time.time()
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype
        ):
            result = self.model.query(encoded, prompt, stream=False)

        query_time = time.time() - query_start
        timings["query_time"] = query_time
        total_time = time.time() - start_time
        timings["total_time"] = total_time

        response = {"answer": result["answer"], "timings": timings}
        # if self.device.type == "xpu": 
        #    torch.xpu.empty_cache()
        return response

    def _query_with_encoded_image(
        self, encoded_image: torch.Tensor, prompt: str
    ) -> Dict:
        """
        Internal method to query with a pre-encoded image tensor.

        Args:
            encoded_image: Pre-encoded image tensor
            prompt: Text prompt to process with the image

        Returns:
            Dictionary with answer and timing information
        """
        if not isinstance(prompt, str):
            prompt = str(prompt)
        start_time = time.time()
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype
        ):
            result = self.model.query(encoded_image, prompt, stream=False)
        query_time = time.time() - start_time

        return {
            "answer": result["answer"],
            "timings": {
                "encode_time": 0.0,  # Already encoded
                "query_time": query_time,
                "total_time": query_time,
            },
        }

    def batch_query(
        self,
        images: List[Union[torch.Tensor, Image.Image, str, bytes, Path]],
        prompts: List[str],
    ) -> List[Dict]:
        """
        Process multiple image-prompt pairs sequentially in batches.

        Note: This method processes images sequentially, not in parallel. It's designed
        for convenience and memory management, not for performance acceleration.

        Args:
            images: List of images or pre-encoded image tensors
            prompts: List of text prompts to process with the images

        Returns:
            List of dictionaries with answers and timing information
        """
        if len(images) != len(prompts):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})"
            )
        tensor_images = []
        tensor_indices = []
        non_tensor_images = []
        non_tensor_prompts = []
        non_tensor_indices = []

        for i, (img, prompt) in enumerate(zip(images, prompts)):
            if isinstance(img, torch.Tensor):
                tensor_images.append(img)
                tensor_indices.append(i)
            else:
                non_tensor_images.append(img)
                non_tensor_prompts.append(prompt)
                non_tensor_indices.append(i)
        non_tensor_results = []
        if non_tensor_images:
            for img, prompt, idx in zip(
                non_tensor_images, non_tensor_prompts, non_tensor_indices
            ):
                try:
                    result = self.query(img, prompt)
                    non_tensor_results.append((idx, result))
                except Exception as e:
                    logger.error(f"Error processing image {idx+1}: {str(e)}")
                    non_tensor_results.append((idx, {"error": str(e)}))
        tensor_results = []
        for tensor_img, idx in zip(tensor_images, tensor_indices):
            tensor_results.append(
                (idx, self._query_with_encoded_image(tensor_img, prompts[idx]))
            )
        all_results = [None] * len(images)
        for idx, result in non_tensor_results:
            all_results[idx] = result
        for idx, result in tensor_results:
            all_results[idx] = result
        return all_results

    def caption(
        self, image: Union[Image.Image, str, bytes, Path], length: str = "normal"
    ) -> Dict[str, Union[str, float]]:
        """
        Generate a caption for an image.

        Args:
            image: PIL Image, file path, or bytes containing the image
            length: Caption length, "short" or "normal" (default: "normal")

        Returns:
            Dictionary with caption and timing information
        """
        start_time = time.time()
        image_pil = self._open_image(image)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype
        ):
            result = self.model.caption(image_pil, length=length)
        total_time = time.time() - start_time
        response = {"caption": result["caption"], "timings": {"total_time": total_time}}
        if self.device.type == "xpu":
            torch.xpu.empty_cache()
        return response

    def detect(
        self, image: Union[Image.Image, str, bytes, Path], object_type: str
    ) -> Dict[str, Any]:
        """
        Detect objects of a specific type in an image.

        Args:
            image: PIL Image, file path, or bytes containing the image
            object_type: Type of object to detect (e.g., "face", "person", "eyes", "dog")

        Returns:
            Dictionary with detected objects and their bounding boxes
        """
        start_time = time.time()
        # Convert to PIL Image if needed
        image_pil = self._open_image(image)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype
        ):
            result = self.model.detect(image_pil, object_type)
        total_time = time.time() - start_time
        response = {
            "objects": result.get("objects", []),
            "count": len(result.get("objects", [])),
            "timings": {"total_time": total_time},
        }
        if self.device.type == "xpu":
            torch.xpu.empty_cache()

        return response

    def point(
        self, image: Union[Image.Image, str, bytes, Path], object_type: str
    ) -> Dict[str, Any]:
        """
        Point to objects of a specific type in an image.

        Args:
            image: PIL Image, file path, or bytes containing the image
            object_type: Type of object to point to (e.g., "person", "dog", "car")

        Returns:
            Dictionary with points indicating object locations
        """
        start_time = time.time()
        image_pil = self._open_image(image)
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=self.dtype
        ):
            result = self.model.point(image_pil, object_type)
        total_time = time.time() - start_time
        response = {
            "points": result.get("points", []),
            "count": len(result.get("points", [])),
            "timings": {"total_time": total_time},
        }
        if self.device.type == "xpu":
            torch.xpu.empty_cache()
        return response

    def batch_caption(
        self,
        images: List[Union[Image.Image, str, bytes, Path]],
        length: str = "normal",
    ) -> List[Dict[str, Any]]:
        """
        Generate captions for multiple images sequentially in batches.

        Note: This method processes images sequentially, not in parallel. It's designed
        for convenience and memory management, not for performance acceleration.

        Args:
            images: List of images to caption
            length: Caption length, "short" or "normal" (default: "normal")

        Returns:
            List of dictionaries with captions and timing information
        """
        return self.batch_process(self.caption, images, length=length)

    def batch_detect(
        self,
        images: List[Union[Image.Image, str, bytes, Path]],
        object_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in multiple images sequentially in batches.

        Note: This method processes images sequentially, not in parallel. It's designed
        for convenience and memory management, not for performance acceleration.

        Args:
            images: List of images to process
            object_type: Type of object to detect

        Returns:
            List of detection results, one for each input image
        """
        return self.batch_process(self.detect, images, object_type)

    def batch_point(
        self,
        images: List[Union[Image.Image, str, bytes, Path]],
        object_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Point to objects in multiple images sequentially in batches.

        Note: This method processes images sequentially, not in parallel. It's designed
        for convenience and memory management, not for performance acceleration.

        Args:
            images: List of images to process
            object_type: Type of object to point to

        Returns:
            List of pointing results, one for each input image
        """
        return self.batch_process(self.point, images, object_type)

    def batch_process(
        self,
        func: Callable,
        images: List[Union[Image.Image, str, bytes, Path]],
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images with a given function sequentially in batches.

        Note: This method processes images sequentially, not in parallel. It's designed
        for convenience and memory management, not for performance acceleration.

        Args:
            func: Function to apply to each image (e.g., self.caption, self.detect)
            images: List of images to process
            *args, **kwargs: Additional arguments to pass to the function

        Returns:
            List of results, one for each input image
        """
        results = []
        batch_size = min(self.max_batch_size, len(images))
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            for idx, image in enumerate(batch):
                batch_idx = i + idx
                logger.debug(f"Processing image {batch_idx+1}/{len(images)}")
                try:
                    if (
                        args
                        and isinstance(args[0], list)
                        and len(args[0]) == len(images)
                    ):
                        arg_for_this_image = args[0][batch_idx]
                        other_args = args[1:]
                        result = func(image, arg_for_this_image, *other_args, **kwargs)
                    else:
                        result = func(image, *args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image {batch_idx+1}: {str(e)}")
                    results.append({"error": str(e)})
            if self.device.type == "xpu" and i + batch_size < len(images):
                torch.xpu.empty_cache()

        return results

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics for the device"""
        if self.device.type == "xpu":
            free_mem, total_mem = torch.xpu.mem_get_info()
            used_mem = total_mem - free_mem
            return {
                "total_memory_gb": total_mem / (1024**3),
                "used_memory_gb": used_mem / (1024**3),
                "free_memory_gb": free_mem / (1024**3),
                "utilization_percent": (used_mem / total_mem) * 100,
            }
        else:
            return {
                "total_memory_gb": 0,
                "used_memory_gb": 0,
                "free_memory_gb": 0,
                "utilization_percent": 0,
            }

    def __del__(self):
        """Clean up resources when the engine is destroyed"""
        if hasattr(self, "initialized") and self.initialized:
            if self.device.type == "xpu":
                torch.xpu.empty_cache()
                logger.info("XPU cache cleared")


if __name__ == "__main__":
    """
    Run a self-test of the MoondreamEngine.
    """
    TEST_IMAGES = {
        "portrait": "https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg",
        "dog": "https://images.unsplash.com/photo-1530281700549-e82e7bf110d6?q=80&w=1000",
        "nature": "https://images.pexels.com/photos/268533/pexels-photo-268533.jpeg",
        "multi_person": "https://media.cntraveler.com/photos/607f3c487774091e06dd5d21/16:9/w_2560%2Cc_limit/Outdoor%2520Dining_G8C6XK.jpg",
        "dogs": "https://www.akc.org/wp-content/uploads/2017/11/Labrador-Retrievers-three-colors.jpg",
    }

    def fetch_image(url):
        """Utility function to fetch an image from a URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to fetch image from {url}: {e}")
            return None

    def run_tests():
        """Run a comprehensive test of all engine functionality"""
        logger.info("Starting MoondreamEngine self-test")
        engine = MoondreamEngine(revision=None)

        # Fetch all test images
        images = {}
        for name, url in TEST_IMAGES.items():
            img = fetch_image(url)
            if img:
                images[name] = img

        if not images:
            logger.error("Failed to fetch any test images. Aborting test.")
            return

        # 1. Test single operations
        logger.info("\n=== Testing Single Operations ===")

        # Caption
        logger.info("\nTesting caption functionality")
        caption_result = engine.caption(images["portrait"])
        logger.info(f"Caption: {caption_result['caption']}")
        logger.info(f"Time: {caption_result['timings']['total_time']:.4f}s")

        # Query
        logger.info("\nTesting query functionality")
        query_result = engine.query(
            images["dog"], "What is the relationship between the woman and dog?"
        )
        logger.info(f"Answer: {query_result['answer']}")
        logger.info(f"Time: {query_result['timings']['total_time']:.4f}s")

        # Detect
        logger.info("\nTesting detect functionality")
        detect_result = engine.detect(images["dogs"], "dog")
        logger.info(f"Detected {detect_result['count']} dogs")
        logger.info(f"Time: {detect_result['timings']['total_time']:.4f}s")

        # Point
        logger.info("\nTesting point functionality")
        point_result = engine.point(images["multi_person"], "person")
        logger.info(f"Found {point_result['count']} people")
        logger.info(f"Time: {point_result['timings']['total_time']:.4f}s")

        # 2. Test batch operations
        logger.info("\n=== Testing Batch Operations ===")

        # Prepare batch inputs
        batch_images = list(images.values())[:3]  # Use first 3 images

        # Batch Caption
        logger.info("\nTesting batch_caption")
        batch_caption_results = engine.batch_caption(batch_images)
        for i, result in enumerate(batch_caption_results):
            logger.info(f"Image {i+1} caption: {result['caption']}")

        # Batch Query
        logger.info("\nTesting batch_query")
        batch_prompts = ["What is in this image?"] * len(batch_images)
        batch_query_results = engine.batch_query(batch_images, batch_prompts)
        for i, result in enumerate(batch_query_results):
            if "error" in result:
                logger.error(f"Image {i+1} error: {result['error']}")
            else:
                logger.info(f"Image {i+1} answer: {result['answer']}")

        # Batch Detect
        logger.info("\nTesting batch_detect")
        batch_detect_results = engine.batch_detect(batch_images, "person")
        for i, result in enumerate(batch_detect_results):
            logger.info(f"Image {i+1} detected {result.get('count', 0)} people")

        # Batch Point
        logger.info("\nTesting batch_point")
        batch_point_results = engine.batch_point(batch_images, "person")
        for i, result in enumerate(batch_point_results):
            logger.info(f"Image {i+1} found {result.get('count', 0)} people")

        # Generic batch_process
        logger.info("\nTesting generic batch_process")
        batch_results = engine.batch_process(engine.caption, batch_images[:2])
        for i, result in enumerate(batch_results):
            logger.info(
                f"Batch process result {i+1}: {result.get('caption', result.get('error', 'Unknown'))}"
            )

        logger.info("\nSelf-test completed!")

    run_tests()  # run all tests if this file is run directly to test the backend
