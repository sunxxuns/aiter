#!/usr/bin/env python3
"""Simple test for fwd_fp8_256t_bf16style.s kernel"""
import os
os.environ.setdefault('HIP_VISIBLE_DEVICES', '7')

import torch
import ctypes
import struct

# Load HIP runtime
hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")

# Define HIP types
hipModule_t = ctypes.c_void_p
hipFunction_t = ctypes.c_void_p

def load_module(path):
    """Load a code object file"""
    module = hipModule_t()
    ret = hip.hipModuleLoad(ctypes.byref(module), path.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    return module

def get_function(module, name):
    """Get kernel function from module"""
    func = hipFunction_t()
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, name.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    return func

def launch_kernel(func, grid, block, args_ptr, shared_mem=0, stream=None):
    """Launch kernel"""
    ret = hip.hipModuleLaunchKernel(
        func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        shared_mem,
        stream,
        args_ptr,
        None
    )
    if ret != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed: {ret}")

KERNEL_NAME = "_ZN5aiter19fwd_fp8_256t_bf16stE"

def test_kernel(seq_len=64):
    """Test the 256T BF16-style kernel"""
    print("Entered test_kernel", flush=True)
    hd = 128
    Q_rows = 128  # Fixed for 256T kernel (4 waves × 32 rows)
    
    print(f"Testing seq_len={seq_len}, Q_rows={Q_rows}, hd={hd}", flush=True)
    
    # Create tensors
    print("Creating Q...", flush=True)
    Q = torch.randn(Q_rows, hd, dtype=torch.float32, device='cuda')
    print("Q created", flush=True)
    print("Creating K...", flush=True)
    K = torch.randn(seq_len, hd, dtype=torch.float32, device='cuda')
    print("Creating V...", flush=True)
    V = torch.randn(seq_len, hd, dtype=torch.float32, device='cuda')
    print("Creating O...", flush=True)
    O = torch.zeros(Q_rows, hd, dtype=torch.float32, device='cuda')
    print("Tensors created", flush=True)
    
    # Convert to FP8
    print("Converting to FP8...", flush=True)
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    print("Q converted", flush=True)
    print("Converting K...", flush=True)
    K_fp8 = K.to(torch.float8_e4m3fn)
    print("Converting V...", flush=True)
    V_fp8 = V.to(torch.float8_e4m3fn)
    print("All FP8 conversions done", flush=True)
    
    # Load kernel
    co_path = os.path.join(os.path.dirname(__file__), "fwd_fp8_256t_bf16style.co")
    print(f"Loading: {co_path}", flush=True)
    print(f"File exists: {os.path.exists(co_path)}", flush=True)
    module = load_module(co_path)
    print(f"Module loaded: {module}", flush=True)
    kernel = get_function(module, KERNEL_NAME)
    print(f"Kernel loaded: {kernel}", flush=True)
    
    # Q_row_stride in bytes (FP8 = 1 byte per element)
    Q_row_stride = hd  # 128 bytes per row
    print(f"Q_row_stride: {Q_row_stride}", flush=True)
    
    # Pack kernel args: O*, Q*, K*, V*, seq_len, Q_row_stride
    print("Packing args...", flush=True)
    print(f"  O ptr: {O.data_ptr()}", flush=True)
    print(f"  Q ptr: {Q_fp8.data_ptr()}", flush=True)
    print(f"  K ptr: {K_fp8.data_ptr()}", flush=True)
    print(f"  V ptr: {V_fp8.data_ptr()}", flush=True)
    args = struct.pack('<QQQQII',
        O.data_ptr(),
        Q_fp8.data_ptr(),
        K_fp8.data_ptr(),
        V_fp8.data_ptr(),
        seq_len,
        Q_row_stride
    )
    print(f"Args packed: {len(args)} bytes", flush=True)
    
    # Create args buffer on HOST (not GPU!)
    print("Creating args buffer on host...", flush=True)
    args_buffer = ctypes.create_string_buffer(args)
    args_ptr = ctypes.cast(args_buffer, ctypes.c_void_p).value
    print(f"Args ptr: {args_ptr}", flush=True)
    
    # Create kernel args array (pointer to the args buffer)
    kernel_args = (ctypes.c_void_p * 1)(args_ptr)
    print(f"Kernel args ready", flush=True)
    
    # Launch: 1 block of 256 threads
    grid = (1, 1, 1)
    block = (256, 1, 1)
    shared_mem = 65536  # 64KB LDS
    
    print(f"About to launch: grid={grid}, block={block}, shared={shared_mem}", flush=True)
    print("Calling launch_kernel...", flush=True)
    launch_kernel(kernel, grid, block, kernel_args, shared_mem)
    print("Launch returned", flush=True)
    
    # Sync
    torch.cuda.synchronize()
    
    # Check output
    print(f"Results:")
    print(f"  O mean: {O.mean().item():.6f}")
    print(f"  O std:  {O.std().item():.6f}")
    print(f"  O nan:  {torch.isnan(O).sum().item()}")
    print(f"  O[0,:4]: {O[0,:4].tolist()}")
    
    # Reference
    scale = 1.0 / (hd ** 0.5)
    S = torch.matmul(Q.float(), K.float().T) * scale
    P = torch.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V.float())
    
    diff = (O - O_ref).abs()
    print(f"  Max diff: {diff.max().item():.4f}")
    
    return O, O_ref

if __name__ == "__main__":
    import sys
    print("Starting...", flush=True)
    print(f"Python: {sys.version}", flush=True)
    print("=" * 60, flush=True)
    print("Testing fwd_fp8_256t_bf16style.s", flush=True)
    print("=" * 60, flush=True)
    
    print("About to call test_kernel...", flush=True)
    test_kernel(seq_len=32)
