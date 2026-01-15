#!/usr/bin/env python3
"""
Test diagonal padding kernel for correctness and performance.
"""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def test_kernel(name, co_file, func_name, seq_len=64):
    """Test kernel for correctness."""
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    if not os.path.exists(co_file):
        print(f"Kernel not found: {co_file}")
        return None
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name.encode())
    
    # All ones input
    Q = torch.ones(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.ones(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Expected: 128 * num_k_tiles
    num_k_tiles = (seq_len + 31) // 32
    expected = 128 * num_k_tiles
    
    result = {
        'name': name,
        'expected': expected,
        'got': O[0].item(),
        'mean': O[:1024].mean().item(),
        'correct': abs(O[0].item() - expected) < 0.1,
    }
    
    hip.hipModuleUnload(module)
    return result

def benchmark_kernel(name, co_file, func_name, seq_len=32128, num_heads=40):
    """Benchmark kernel performance."""
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    if not os.path.exists(co_file):
        return None
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name.encode())
    
    Q = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
    
    num_blocks = ((seq_len + 31) // 32) * num_heads
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Warmup
    for _ in range(3):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    iterations = 10
    start.record()
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iterations
    
    # TF/s calculation
    flops_per_block = 2 * 32 * seq_len * 128
    total_flops = flops_per_block * num_blocks
    tflops = total_flops / (time_ms / 1000) / 1e12
    
    hip.hipModuleUnload(module)
    
    return {'name': name, 'time_ms': time_ms, 'tflops': tflops}

def main():
    print("=" * 70)
    print("DIAGONAL PADDING KERNEL TEST")
    print("=" * 70)
    
    kernels = [
        ("preload (baseline)", 
         "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co",
         "_ZN5aiter17fwd_fp8_qk_preloadE"),
        ("diag_pad (new)", 
         "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_diag_pad.co",
         "_ZN5aiter18fwd_fp8_qk_diag_padE"),
    ]
    
    # Correctness test
    print("\n--- Correctness Test (all-ones, seq_len=64) ---")
    print(f"{'Kernel':<25} | {'Expected':>10} | {'Got':>10} | {'Correct':>8}")
    print("-" * 60)
    
    for name, co_file, func_name in kernels:
        result = test_kernel(name, co_file, func_name, seq_len=64)
        if result:
            status = "PASS" if result['correct'] else "FAIL"
            print(f"{result['name']:<25} | {result['expected']:>10.1f} | {result['got']:>10.1f} | {status:>8}")
    
    # Performance test
    print("\n--- Performance Test (seq_len=32128, heads=40) ---")
    print(f"{'Kernel':<25} | {'Time (ms)':>10} | {'TF/s':>10}")
    print("-" * 50)
    
    for name, co_file, func_name in kernels:
        result = benchmark_kernel(name, co_file, func_name)
        if result:
            print(f"{result['name']:<25} | {result['time_ms']:>10.3f} | {result['tflops']:>10.1f}")
    
    # Rocprof comparison
    print("\n--- Running rocprof for bank conflict analysis ---")
    import subprocess
    
    # Create a test script for rocprof
    bench_script = '''
import torch
import ctypes
import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

kernels = [
    ("preload", "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co", "_ZN5aiter17fwd_fp8_qk_preloadE"),
    ("diag_pad", "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_diag_pad.co", "_ZN5aiter18fwd_fp8_qk_diag_padE"),
]

for name, co_file, func_name in kernels:
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name.encode())
    
    Q = torch.ones(64, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.ones(64, 128, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(O.data_ptr()), ctypes.c_void_p(Q.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_uint32(64), ctypes.c_uint32(0)]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)
    print(f"Ran {name}")
'''
    
    with open("/tmp/rocprof_diag.py", 'w') as f:
        f.write(bench_script)
    
    cmd = "echo 'pmc: SQ_LDS_BANK_CONFLICT SQ_WAVES' > /tmp/rocprof_diag_input.txt && rocprof -i /tmp/rocprof_diag_input.txt -o /tmp/rocprof_diag_output.csv python /tmp/rocprof_diag.py 2>&1 | tail -20"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Show rocprof results
    cmd = "cat /tmp/rocprof_diag_output.csv | grep -E 'KernelName|fwd_fp8'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("\nRocprof Results:")
    print(result.stdout)

if __name__ == "__main__":
    main()
