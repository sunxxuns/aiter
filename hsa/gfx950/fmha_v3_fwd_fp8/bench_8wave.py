import torch, ctypes, os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

# Load kernels
module_4w = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module_4w), b'fwd_fp8_qk_preload.co')
func_4w = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func_4w), module_4w, b'_ZN5aiter17fwd_fp8_qk_preloadE')

module_8w = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module_8w), b'fwd_fp8_qk_8wave.co')
func_8w = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func_8w), module_8w, b'_ZN5aiter15fwd_fp8_qk_8waveE')

HEADS = 40

def bench_4wave(seq, iters=10):
    S = torch.zeros(64, 16, dtype=torch.float32, device='cuda')
    Q = torch.randn(HEADS, seq, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(HEADS, seq, 128, device='cuda').to(torch.float8_e4m3fn)
    num_blocks = HEADS * (seq // 32)  # 32 rows per block
    
    args = [ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
            ctypes.c_void_p(K.data_ptr()), ctypes.c_uint32(seq), ctypes.c_uint32(0)]
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    for _ in range(3):
        hip.hipModuleLaunchKernel(func_4w, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func_4w, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    end.record()
    end.synchronize()
    
    time_ms = start.elapsed_time(end) / iters
    flops = 2 * HEADS * seq * seq * 128
    return flops / (time_ms * 1e9)

def bench_8wave(seq, iters=10):
    S = torch.zeros(128, 16, dtype=torch.float32, device='cuda')  # 64 rows output
    Q = torch.randn(HEADS, seq, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(HEADS, seq, 128, device='cuda').to(torch.float8_e4m3fn)
    num_blocks = HEADS * (seq // 64)  # 64 rows per block with 8 waves
    
    args = [ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
            ctypes.c_void_p(K.data_ptr()), ctypes.c_uint32(seq), ctypes.c_uint32(0)]
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    for _ in range(3):
        hip.hipModuleLaunchKernel(func_8w, num_blocks, 1, 1, 512, 1, 1, 20480, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func_8w, num_blocks, 1, 1, 512, 1, 1, 20480, None, args_ptrs, None)
    end.record()
    end.synchronize()
    
    time_ms = start.elapsed_time(end) / iters
    flops = 2 * HEADS * seq * seq * 128
    return flops / (time_ms * 1e9)

print(f"{'SEQ':>6} | {'4-wave':>10} | {'8-wave':>10} | {'Speedup':>8}")
print("-" * 48)

for seq in [1024, 2048, 4096, 8192, 16384]:
    tf_4w = bench_4wave(seq)
    tf_8w = bench_8wave(seq)
    speedup = tf_8w / tf_4w
    print(f"{seq:>6} | {tf_4w:>8.1f} | {tf_8w:>8.1f} | {speedup:>7.2f}x")
