"""
Benchmark FP8 Flash Attention kernel
"""
import ctypes
import struct
import torch
import time
import numpy as np

# Load HIP runtime
libhip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')

hipModule_t = ctypes.c_void_p
hipFunction_t = ctypes.c_void_p
hipEvent_t = ctypes.c_void_p

# Function signatures
libhip.hipModuleLoad.argtypes = [ctypes.POINTER(hipModule_t), ctypes.c_char_p]
libhip.hipModuleLoad.restype = ctypes.c_int
libhip.hipModuleGetFunction.argtypes = [ctypes.POINTER(hipFunction_t), hipModule_t, ctypes.c_char_p]
libhip.hipModuleGetFunction.restype = ctypes.c_int
libhip.hipModuleLaunchKernel.argtypes = [
    hipFunction_t, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
libhip.hipModuleLaunchKernel.restype = ctypes.c_int
libhip.hipDeviceSynchronize.argtypes = []
libhip.hipDeviceSynchronize.restype = ctypes.c_int
libhip.hipEventCreate.argtypes = [ctypes.POINTER(hipEvent_t)]
libhip.hipEventCreate.restype = ctypes.c_int
libhip.hipEventRecord.argtypes = [hipEvent_t, ctypes.c_void_p]
libhip.hipEventRecord.restype = ctypes.c_int
libhip.hipEventSynchronize.argtypes = [hipEvent_t]
libhip.hipEventSynchronize.restype = ctypes.c_int
libhip.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), hipEvent_t, hipEvent_t]
libhip.hipEventElapsedTime.restype = ctypes.c_int
libhip.hipEventDestroy.argtypes = [hipEvent_t]
libhip.hipEventDestroy.restype = ctypes.c_int

def pack_args(q, k, v, out, lse, scale, seqlen_q, seqlen_k, q_s, k_s, v_s):
    args = bytearray(528)
    struct.pack_into('Q', args, 0, out.data_ptr())
    struct.pack_into('Q', args, 16, q.data_ptr())
    struct.pack_into('Q', args, 32, k.data_ptr())
    struct.pack_into('Q', args, 48, v.data_ptr())
    struct.pack_into('Q', args, 64, lse.data_ptr())
    struct.pack_into('f', args, 80, scale)
    struct.pack_into('I', args, 88, seqlen_q)
    struct.pack_into('I', args, 96, seqlen_k)
    struct.pack_into('f', args, 512, q_s)
    struct.pack_into('f', args, 516, k_s)
    struct.pack_into('f', args, 520, v_s)
    return bytes(args)

def benchmark_fp8_kernel(seq_q=64, seq_k=64, head_dim=128, warmup=10, iterations=100):
    """Benchmark the FP8 flash attention kernel"""
    
    # Load kernel
    module = hipModule_t()
    ret = libhip.hipModuleLoad(ctypes.byref(module), b'fwd_hd128_fp8.co')
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    
    function = hipFunction_t()
    ret = libhip.hipModuleGetFunction(ctypes.byref(function), module, b'_ZN5aiter18fmha_fwd_hd128_fp8E')
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    
    # Create tensors (FP8 as uint8)
    q = torch.full((seq_q * head_dim,), 0x38, dtype=torch.uint8, device='cuda')
    k = torch.full((seq_k * head_dim,), 0x38, dtype=torch.uint8, device='cuda')
    v = torch.full((seq_k * head_dim,), 0x38, dtype=torch.uint8, device='cuda')
    out = torch.zeros(65536, dtype=torch.float32, device='cuda')
    lse = torch.zeros(1, 1, seq_q, dtype=torch.float32, device='cuda')
    
    scale = 1.0 / (head_dim ** 0.5)
    args = pack_args(q, k, v, out, lse, scale, seq_q, seq_k, 1.0, 1.0, 1.0)
    args_ptr = ctypes.cast(args, ctypes.c_void_p)
    args_size = ctypes.c_size_t(len(args))
    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(0x01), args_ptr,
        ctypes.c_void_p(0x02), ctypes.cast(ctypes.pointer(args_size), ctypes.c_void_p),
        ctypes.c_void_p(0x03)
    )
    
    # Create events
    start_event = hipEvent_t()
    end_event = hipEvent_t()
    libhip.hipEventCreate(ctypes.byref(start_event))
    libhip.hipEventCreate(ctypes.byref(end_event))
    
    # Warmup
    for _ in range(warmup):
        libhip.hipModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 32768, None, None, ctypes.cast(extra, ctypes.c_void_p))
    libhip.hipDeviceSynchronize()
    
    # Benchmark with events
    libhip.hipEventRecord(start_event, None)
    for _ in range(iterations):
        libhip.hipModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 32768, None, None, ctypes.cast(extra, ctypes.c_void_p))
    libhip.hipEventRecord(end_event, None)
    libhip.hipEventSynchronize(end_event)
    
    elapsed_ms = ctypes.c_float()
    libhip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, end_event)
    
    avg_time_ms = elapsed_ms.value / iterations
    
    # Calculate TFLOPS
    # FLOPs = 2 * seq_q * seq_k * head_dim (for QK^T) + 2 * seq_q * seq_k * head_dim (for PV)
    flops = 2.0 * seq_q * seq_k * head_dim * 2  # QK and PV
    tflops = flops / (avg_time_ms * 1e9)
    
    # Cleanup
    libhip.hipEventDestroy(start_event)
    libhip.hipEventDestroy(end_event)
    
    return avg_time_ms, tflops

def main():
    print("=" * 70)
    print("FP8 Flash Attention Kernel Benchmark")
    print("=" * 70)
    print(f"Note: Current kernel only handles single tile (seq_q=32, seq_k=32)")
    print(f"      This is a WIP kernel - performance will improve")
    print("=" * 70)
    
    # Current kernel only handles 64 threads = 32x32 output tile
    seq_q, seq_k = 32, 32
    head_dim = 128
    
    print(f"\nConfiguration: seq_q={seq_q}, seq_k={seq_k}, head_dim={head_dim}")
    print(f"{'Warmup':>10} {'Iters':>10} {'Time(us)':>12} {'TFLOPS':>12}")
    print("-" * 50)
    
    for warmup, iters in [(10, 100), (50, 500), (100, 1000)]:
        try:
            time_ms, tflops = benchmark_fp8_kernel(seq_q, seq_k, head_dim, warmup, iters)
            time_us = time_ms * 1000
            print(f"{warmup:>10} {iters:>10} {time_us:>12.2f} {tflops:>12.2f}")
        except Exception as e:
            print(f"{warmup:>10} {iters:>10} ERROR: {e}")
    
    print("=" * 70)
    print("\nReference: AMD MI300X theoretical peak:")
    print("  - FP8 TFLOPS: ~2600 TFLOPS (matrix)")
    print("  - BF16 TFLOPS: ~1300 TFLOPS (matrix)")
    print("  - Target: FP8 should be ~2x BF16 performance")

if __name__ == "__main__":
    main()
