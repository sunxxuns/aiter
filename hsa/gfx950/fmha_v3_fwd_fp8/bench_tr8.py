import torch, ctypes, os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

# Load TR8 kernel
module = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module), b'fwd_fp8_qk_v8swizzle.co')
func = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func), module, b'fwd_fp8_qk_v8swizzle')

# Also load preload kernel for comparison
module2 = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module2), b'fwd_fp8_qk_preload.co')
func_preload = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func_preload), module2, b'_ZN5aiter17fwd_fp8_qk_preloadE')

def bench_tr8(num_k_tiles, iters=20):
    """TR8 kernel: single block, processes num_k_tiles"""
    O = torch.zeros(256, 16, dtype=torch.float32, device='cuda')
    Q = torch.randn(32, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(num_k_tiles * 32, 128, device='cuda').to(torch.float8_e4m3fn)
    
    args = [ctypes.c_void_p(O.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
            ctypes.c_void_p(K.data_ptr()), ctypes.c_uint32(num_k_tiles)]
    args_ptrs = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    for _ in range(5):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    end.record()
    end.synchronize()
    
    time_us = start.elapsed_time(end) / iters * 1000
    flops = 2 * 32 * (num_k_tiles * 32) * 128  # QK for 32 Q rows
    tfs = flops / (time_us * 1e6) / 1e12
    return time_us, tfs

def bench_preload(seq, iters=10):
    """Preload kernel: multi-block"""
    S = torch.zeros(64, 16, dtype=torch.float32, device='cuda')
    Q = torch.randn(1, seq, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(1, seq, 128, device='cuda').to(torch.float8_e4m3fn)
    num_blocks = seq // 32
    
    args = [ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
            ctypes.c_void_p(K.data_ptr()), ctypes.c_uint32(seq), ctypes.c_uint32(0)]
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    for _ in range(3):
        hip.hipModuleLaunchKernel(func_preload, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func_preload, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    end.record()
    end.synchronize()
    
    time_us = start.elapsed_time(end) / iters * 1000
    flops = 2 * seq * seq * 128
    tfs = flops / (time_us * 1e6) / 1e12
    return time_us, tfs

print("=== TR8 Kernel (single block) ===")
print(f"{'K_tiles':>8} | {'Time(us)':>10} | {'TF/s':>10}")
print("-" * 35)
for k_tiles in [4, 8, 16, 32, 64, 128]:
    time_us, tfs = bench_tr8(k_tiles)
    print(f"{k_tiles:>8} | {time_us:>10.2f} | {tfs*1000:>10.1f}")

print("\n=== Preload Kernel (multi-block, H=1) ===")
print(f"{'SEQ':>8} | {'Time(us)':>10} | {'TF/s':>10}")
print("-" * 35)
for seq in [128, 256, 512, 1024, 2048, 4096]:
    time_us, tfs = bench_preload(seq)
    print(f"{seq:>8} | {time_us:>10.2f} | {tfs*1000:>10.1f}")
