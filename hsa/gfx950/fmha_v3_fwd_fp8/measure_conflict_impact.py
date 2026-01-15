#!/usr/bin/env python3
"""
Measure the actual performance impact of bank conflicts.
"""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def measure_kernel_time():
    """Measure precise kernel execution time."""
    
    co_file = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co"
    func_name = b"_ZN5aiter17fwd_fp8_qk_preloadE"
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name)
    
    seq_len = 64
    
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
    
    # Warmup
    for _ in range(10):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    iterations = 1000
    start.record()
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end)
    time_per_kernel_us = (time_ms / iterations) * 1000
    
    hip.hipModuleUnload(module)
    
    return time_per_kernel_us

def main():
    print("=" * 70)
    print("BANK CONFLICT IMPACT ANALYSIS")
    print("=" * 70)
    
    time_us = measure_kernel_time()
    
    print(f"\nKernel execution time: {time_us:.3f} us")
    
    # From rocprof:
    bank_conflict_cycles = 2048
    
    # GPU clock frequency for gfx950 (MI300X): ~1.7 GHz typical
    # LDS operates at this frequency
    gpu_freq_ghz = 1.7
    
    conflict_time_us = bank_conflict_cycles / (gpu_freq_ghz * 1000)
    
    print(f"Bank conflict cycles: {bank_conflict_cycles}")
    print(f"Estimated conflict time: {conflict_time_us:.3f} us (at {gpu_freq_ghz} GHz)")
    print(f"Conflict time as % of kernel: {conflict_time_us / time_us * 100:.1f}%")
    
    # Calculate potential speedup from eliminating conflicts
    potential_speedup = time_us / (time_us - conflict_time_us)
    
    print(f"\nPotential speedup from zero conflicts: {potential_speedup:.3f}x ({(potential_speedup-1)*100:.1f}% faster)")
    
    # Context: What matters for overall performance?
    print("\n" + "=" * 70)
    print("CONTEXT: What dominates kernel time?")
    print("=" * 70)
    
    # MFMA throughput dominates for compute-bound kernels
    # For 32x32x16 MFMA FP8: 32*32*16*2 = 32768 FP8 ops per instruction
    # At 1 MFMA per 16 cycles (estimate), for 1 CU:
    # 32768 ops / 16 cycles * 1.7 GHz = 3.5 TFLOP/s per CU
    
    # Our kernel does:
    # - Q load from global to LDS (memory)
    # - K load from global to LDS (memory)  
    # - ds_read for Q (LDS)
    # - ds_read for K (LDS)
    # - MFMA (compute)
    # - Store result (memory)
    
    print("""
Kernel components:
1. Global memory loads (Q, K): ~10-50 us for large seq_len
2. LDS reads (Q, K): ~1-2 us
3. MFMA compute: ~0.5-2 us  
4. Global memory store (O): ~1-5 us

For small kernel (seq_len=64):
- Total time: ~{:.1f} us
- Bank conflict overhead: ~{:.1f} us ({:.1f}%)

For large kernel (seq_len=32k):
- Memory dominates, conflicts less significant
""".format(time_us, conflict_time_us, conflict_time_us/time_us*100))

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    if conflict_time_us / time_us < 0.05:
        print("""
Bank conflicts are <5% of kernel time.
-> Keep pitch-136 for simplicity
-> Focus optimization on memory and MFMA scheduling
""")
    else:
        print("""
Bank conflicts are significant (>{:.1f}% of kernel time).
-> Consider implementing XOR swizzle
-> Formula: (row * 144 + k_off) XOR (row * 4)
-> Requires matching swizzle in both write and read paths
""".format(conflict_time_us/time_us*100))

if __name__ == "__main__":
    main()
