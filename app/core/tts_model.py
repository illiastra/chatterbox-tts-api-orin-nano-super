"""
TTS model initialization and management
"""

import os
import gc
import asyncio
import torch
from enum import Enum
from typing import Optional, Dict, Any
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from app.core.mtl import SUPPORTED_LANGUAGES
from app.config import Config, detect_device

# Global model instance
_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_is_multilingual = None
_supported_languages = {}
_warmup_completed = False
_is_fp16 = False
_is_low_memory_mode = False


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


def _clear_gpu_memory():
    """Aggressively clear GPU memory before model loading."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        # Reset peak memory stats for monitoring
        torch.cuda.reset_peak_memory_stats()
        print("✓ GPU memory cleared")


def _convert_model_to_fp16(model, on_device='cuda'):
    """
    Convert model components to FP16 for reduced memory usage.

    Note: The tokenizer component must stay FP32 because it uses STFT
    (torch.stft) for mel spectrogram computation, and cuFFT requires FP32
    for non-power-of-two sizes.

    Args:
        model: The model to convert
        on_device: Target device ('cuda' or 'cpu')
    """
    # Convert t3 (LLM transformer) to FP16
    model.t3 = model.t3.half()

    # Convert s3gen components selectively
    # Keep tokenizer in FP32 - required for STFT/FFT operations
    model.s3gen.flow = model.s3gen.flow.half()
    model.s3gen.mel2wav = model.s3gen.mel2wav.half()
    model.s3gen.speaker_encoder = model.s3gen.speaker_encoder.half()

    # Convert voice encoder to FP16
    model.ve = model.ve.half()

    return model


def _move_model_to_device_unified_memory(model, device: str):
    """
    Move model to device optimized for unified memory systems (Jetson).

    On unified memory, CPU and GPU share the same physical RAM. We need to:
    1. Move component to GPU (creates CUDA allocation in shared memory)
    2. Immediately delete CPU reference
    3. Force garbage collection before next component

    This ensures we never hold 2 copies simultaneously.
    """
    import ctypes

    def force_gc():
        """Force aggressive garbage collection"""
        gc.collect()
        gc.collect()
        gc.collect()
        # Try to force Python to release memory back to OS
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    components_order = [
        # Move largest component first while we have the most free memory
        ("t3", "t3 (transformer)", None),
        ("s3gen.flow", "s3gen.flow", "s3gen"),
        ("s3gen.tokenizer", "s3gen.tokenizer", "s3gen"),
        ("s3gen.mel2wav", "s3gen.mel2wav", "s3gen"),
        ("s3gen.speaker_encoder", "s3gen.speaker_encoder", "s3gen"),
        ("ve", "ve (voice encoder)", None),
    ]

    for attr_path, name, parent in components_order:
        print(f"    Moving {name}...")
        force_gc()

        # Get the component
        if parent:
            parent_obj = getattr(model, parent)
            attr_name = attr_path.split('.')[-1]
            component = getattr(parent_obj, attr_name)
        else:
            attr_name = attr_path
            parent_obj = model
            component = getattr(model, attr_path)

        # Move to device
        moved = component.to(device)

        # Critical: overwrite the reference to allow GC of CPU copy
        setattr(parent_obj, attr_name, moved)

        # Explicitly delete local references
        del component
        del moved

        # Force GC to reclaim CPU memory before next component
        force_gc()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"      GPU: {allocated:.1f}MB")

    # CRITICAL: Move registered buffers that are directly on parent modules (not subcomponents)
    # The s3gen.trim_fade buffer is registered on s3gen itself, not on a subcomponent
    print(f"    Moving parent module buffers...")
    for name, buf in model.s3gen.named_buffers(recurse=False):
        if buf.device.type != device:
            # Move this buffer to the target device
            setattr(model.s3gen, name, buf.to(device))
            print(f"      Moved s3gen.{name} to {device}")

    # Update device attribute on model and ALL subcomponents
    # This is critical - each component uses its own device attribute to create tensors
    device_attrs_to_update = [
        model,  # model.device
    ]

    # Add subcomponents that have device attributes
    if hasattr(model, 't3'):
        device_attrs_to_update.append(model.t3)
    if hasattr(model, 's3gen'):
        device_attrs_to_update.append(model.s3gen)
        if hasattr(model.s3gen, 'tokenizer'):
            device_attrs_to_update.append(model.s3gen.tokenizer)
        if hasattr(model.s3gen, 'mel2wav'):
            device_attrs_to_update.append(model.s3gen.mel2wav)
        if hasattr(model.s3gen, 'flow'):
            device_attrs_to_update.append(model.s3gen.flow)
        if hasattr(model.s3gen, 'speaker_encoder'):
            device_attrs_to_update.append(model.s3gen.speaker_encoder)
    if hasattr(model, 've'):
        device_attrs_to_update.append(model.ve)

    for component in device_attrs_to_update:
        if hasattr(component, 'device'):
            try:
                component.device = device
            except AttributeError:
                # Some components may have read-only device property
                pass

    return model


def _move_model_to_device(model, device: str):
    """
    Move all model components to the specified device one at a time.
    ChatterboxTTS doesn't have a .to() method, so we move components individually.
    """
    # Use unified memory optimized path for CUDA
    if device == 'cuda':
        return _move_model_to_device_unified_memory(model, device)

    # Standard path for CPU
    components = [
        ("t3", None),
        ("s3gen.flow", "s3gen"),
        ("s3gen.tokenizer", "s3gen"),
        ("s3gen.mel2wav", "s3gen"),
        ("s3gen.speaker_encoder", "s3gen"),
        ("ve", None),
    ]

    for attr_path, parent in components:
        if parent:
            parent_obj = getattr(model, parent)
            attr_name = attr_path.split('.')[-1]
            setattr(parent_obj, attr_name, getattr(parent_obj, attr_name).to(device))
        else:
            setattr(model, attr_path, getattr(model, attr_path).to(device))

    if hasattr(model, 'device'):
        model.device = device

    return model


def _is_jetson():
    """Detect if running on NVIDIA Jetson (unified memory system)."""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            return True
    except FileNotFoundError:
        pass
    # Alternative check
    try:
        import subprocess
        result = subprocess.run(['cat', '/proc/device-tree/model'], capture_output=True, text=True)
        if 'NVIDIA' in result.stdout and ('Jetson' in result.stdout or 'Orin' in result.stdout):
            return True
    except:
        pass
    return False


def _load_model_low_memory(use_multilingual: bool):
    """
    Load model in low-memory mode for constrained devices like Jetson.

    On Jetson (unified memory), we load directly to CUDA to avoid the
    CPU->GPU copy which would temporarily require 2x memory.

    On discrete GPU systems, we load to CPU first, convert to FP16,
    then move piece by piece.
    """
    import ctypes

    is_jetson = _is_jetson()

    if is_jetson:
        print("📦 Jetson detected: Loading directly to CUDA (unified memory)...")
    else:
        print("📦 Low memory mode: Loading model on CPU first...")

    # Step 1: Aggressive memory cleanup
    gc.collect()
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass
    _clear_gpu_memory()

    if is_jetson:
        # On Jetson unified memory: load on CPU, convert to FP16, move to CUDA
        print("  Configuring for Jetson unified memory...")

        # Load model on CPU first (shares unified memory anyway)
        if use_multilingual:
            print("  Loading Multilingual model on CPU...")
            model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
        else:
            print("  Loading Standard model on CPU...")
            model = ChatterboxTTS.from_pretrained(device='cpu')

        # Force garbage collection before conversion
        gc.collect()
        gc.collect()

        # Convert to FP16 on CPU (halves memory footprint)
        print("  Converting to FP16 on CPU...")
        model = _convert_model_to_fp16(model, on_device='cpu')

        # Aggressively free memory
        gc.collect()
        gc.collect()
        gc.collect()
        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass

        # Move to CUDA
        print("  Moving model to CUDA...")
        _clear_gpu_memory()
        model = _move_model_to_device(model, 'cuda')

        # Tokenizer must stay FP32 for cuFFT, but keep on CUDA
        print("  Converting tokenizer to FP32 (cuFFT requirement)...")
        model.s3gen.tokenizer = model.s3gen.tokenizer.float().to('cuda')

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  GPU memory (FP16): {allocated:.1f}MB")
    else:
        # Standard path for discrete GPUs: load CPU -> FP16 -> move to GPU
        if use_multilingual:
            print("  Loading Multilingual model on CPU...")
            model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
        else:
            print("  Loading Standard model on CPU...")
            model = ChatterboxTTS.from_pretrained(device='cpu')

        print("  Converting to FP16 on CPU...")
        model = _convert_model_to_fp16(model, on_device='cpu')
        gc.collect()

        print("  Moving model to GPU...")
        _clear_gpu_memory()
        model = _move_model_to_device(model, 'cuda')

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"✓ Model loaded in low-memory mode - GPU: {allocated:.1f}MB")

    return model


async def initialize_model():
    """Initialize the Chatterbox TTS model"""
    global _model, _device, _initialization_state, _initialization_error, _initialization_progress, _is_multilingual, _supported_languages, _is_fp16, _is_low_memory_mode

    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."

        Config.validate()
        _device = detect_device()
        _is_low_memory_mode = Config.LOW_MEMORY_MODE

        # Apply CUDA optimizations early
        import torch
        if Config.TORCH_CUDNN_BENCHMARK and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print(f"✓ CUDNN benchmark mode enabled")

        print(f"Initializing Chatterbox TTS model...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")
        if _is_low_memory_mode:
            print(f"⚡ Low memory mode: ENABLED")

        _initialization_progress = "Creating model cache directory..."
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)

        _initialization_progress = "Checking voice sample..."
        # Check voice sample exists
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}")

        _initialization_progress = "Configuring device compatibility..."
        # Patch torch.load for CPU compatibility if needed
        if _device == 'cpu':
            import torch
            original_load = torch.load
            original_load_file = None

            # Try to patch safetensors if available
            try:
                import safetensors.torch
                original_load_file = safetensors.torch.load_file
            except ImportError:
                pass

            def force_cpu_torch_load(f, map_location=None, **kwargs):
                # Always force CPU mapping if we're on a CPU device
                return original_load(f, map_location='cpu', **kwargs)

            def force_cpu_load_file(filename, device=None):
                # Force CPU for safetensors loading too
                return original_load_file(filename, device='cpu')

            torch.load = force_cpu_torch_load
            if original_load_file:
                safetensors.torch.load_file = force_cpu_load_file

        # Determine if we should use multilingual model
        use_multilingual = Config.USE_MULTILINGUAL_MODEL

        # Initialize model with run_in_executor for non-blocking
        loop = asyncio.get_event_loop()

        # Use low-memory loading for constrained devices (Jetson, etc.)
        if _is_low_memory_mode and _device == 'cuda':
            _initialization_progress = "Loading TTS model in low-memory mode..."
            _model = await loop.run_in_executor(
                None,
                lambda: _load_model_low_memory(use_multilingual)
            )
            _is_multilingual = use_multilingual
            _is_fp16 = True  # Low memory mode always uses FP16
            if use_multilingual:
                _supported_languages = SUPPORTED_LANGUAGES.copy()
                print(f"✓ Multilingual model initialized with {len(_supported_languages)} languages")
            else:
                _supported_languages = {"en": "English"}
                print(f"✓ Standard model initialized (English only)")
        else:
            # Standard loading path
            _initialization_progress = "Loading TTS model (this may take a while)..."

            if use_multilingual:
                print(f"Loading Chatterbox Multilingual TTS model...")
                _model = await loop.run_in_executor(
                    None,
                    lambda: ChatterboxMultilingualTTS.from_pretrained(device=_device)
                )
                _is_multilingual = True
                _supported_languages = SUPPORTED_LANGUAGES.copy()
                print(f"✓ Multilingual model initialized with {len(_supported_languages)} languages")
            else:
                print(f"Loading standard Chatterbox TTS model...")
                _model = await loop.run_in_executor(
                    None,
                    lambda: ChatterboxTTS.from_pretrained(device=_device)
                )
                _is_multilingual = False
                _supported_languages = {"en": "English"}  # Standard model only supports English
                print(f"✓ Standard model initialized (English only)")

            # Convert model to FP16 if enabled (only on CUDA devices)
            if Config.USE_FP16 and _device == 'cuda':
                _initialization_progress = "Converting model to FP16..."
                print("Converting model to FP16 for reduced memory usage...")
                _model = _convert_model_to_fp16(_model)
                _is_fp16 = True

                # Log memory after FP16 conversion
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024 / 1024
                    reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    print(f"✓ FP16 conversion complete - GPU memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
            elif Config.USE_FP16 and _device != 'cuda':
                print("⚠️ FP16 disabled: only supported on CUDA devices")
                _is_fp16 = False
            else:
                _is_fp16 = False

        # Run warm-up inference to prime cuDNN benchmark cache
        if Config.WARMUP_ON_STARTUP:
            _initialization_progress = "Running warm-up inference for cuDNN optimization..."
            print("Running warm-up inference to optimize cuDNN kernels...")
            await warmup_model(num_runs=Config.WARMUP_RUNS)
        else:
            print("Warm-up disabled (WARMUP_ON_STARTUP=false)")

        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"✓ Model initialized successfully on {_device}")
        return _model
        
    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize model: {e}")
        raise e


def get_model():
    """Get the current model instance"""
    return _model


def get_device():
    """Get the current device"""
    return _device


def get_initialization_state():
    """Get the current initialization state"""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message"""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization error if any"""
    return _initialization_error


def is_ready():
    """Check if the model is ready for use"""
    return _initialization_state == InitializationState.READY.value and _model is not None


def is_initializing():
    """Check if the model is currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value 


def is_multilingual():
    """Check if the loaded model supports multilingual generation"""
    return _is_multilingual


def is_fp16():
    """Check if the model is running in FP16 mode"""
    return _is_fp16


def get_supported_languages():
    """Get the dictionary of supported languages"""
    return _supported_languages.copy()


def supports_language(language_id: str):
    """Check if the model supports a specific language"""
    return language_id in _supported_languages


def is_low_memory_mode():
    """Check if the model is running in low memory mode"""
    return _is_low_memory_mode


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information"""
    info = {
        "model_type": "multilingual" if _is_multilingual else "standard",
        "is_multilingual": _is_multilingual,
        "supported_languages": _supported_languages,
        "language_count": len(_supported_languages),
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state,
        "warmup_completed": _warmup_completed,
        "is_fp16": _is_fp16,
        "low_memory_mode": _is_low_memory_mode
    }

    # Add memory stats if on CUDA
    if torch.cuda.is_available():
        info["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        info["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)

    return info


def cleanup_idle_memory():
    """
    Clean up GPU memory when idle.
    Call this periodically or after completing requests to minimize memory footprint.
    """
    if torch.cuda.is_available():
        # Clear cached memory that's not currently in use
        torch.cuda.empty_cache()
        gc.collect()
        return True
    return False


async def warmup_model(num_runs: int = 3):
    """
    Run warm-up inferences to prime the cuDNN benchmark cache.

    cuDNN benchmark mode profiles different kernel implementations on the first
    run with each input shape. Running a few warm-up inferences ensures optimal
    kernels are selected before real requests arrive.

    Args:
        num_runs: Number of warm-up inferences to run (default: 3)
    """
    global _warmup_completed

    if _model is None:
        print("⚠️ Cannot warm up: model not loaded")
        return

    if not os.path.exists(Config.VOICE_SAMPLE_PATH):
        print(f"⚠️ Cannot warm up: voice sample not found at {Config.VOICE_SAMPLE_PATH}")
        return

    # Short test phrases of varying lengths to warm up different input shapes
    warmup_phrases = [
        "Hello, this is a warm-up test.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing one two three, warming up the neural network for optimal performance."
    ]

    loop = asyncio.get_event_loop()

    try:
        for i, phrase in enumerate(warmup_phrases[:num_runs]):
            print(f"  Warm-up {i+1}/{num_runs}: '{phrase[:30]}...'")

            with torch.inference_mode():
                # Prepare generation kwargs
                generate_kwargs = {
                    "text": phrase,
                    "audio_prompt_path": Config.VOICE_SAMPLE_PATH,
                    "exaggeration": Config.EXAGGERATION,
                    "cfg_weight": Config.CFG_WEIGHT,
                    "temperature": Config.TEMPERATURE
                }

                # Add language_id for multilingual models
                if _is_multilingual:
                    generate_kwargs["language_id"] = "en"

                # Run inference with autocast for FP16 mode
                def run_generate():
                    if _is_fp16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            return _model.generate(**generate_kwargs)
                    else:
                        return _model.generate(**generate_kwargs)

                # Run inference (non-blocking)
                audio_tensor = await loop.run_in_executor(None, run_generate)

                # Discard output - we only care about warming up the kernels
                del audio_tensor

            # Clear intermediate caches but keep cuDNN benchmark cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _warmup_completed = True
        print(f"✓ Warm-up complete ({num_runs} inferences)")

        # Log GPU state after warm-up
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            print(f"  GPU memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

    except Exception as e:
        print(f"⚠️ Warm-up failed (non-fatal): {e}")
        # Don't fail initialization if warm-up fails
        _warmup_completed = False