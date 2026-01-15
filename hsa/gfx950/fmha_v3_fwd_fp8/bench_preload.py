import torch, ctypes, os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

# Load kernels
kernels = {}
for name, co, sym in [
    ('SwPipe', 'fwd_fp8_qk_swpipe.co', '_ZN5aiter16fwd_fp8_qk_swpipeE'),
    ('Preload', 'fwd_fp8_qk_preload.co', '_ZN5aiter17fwd_fp8_qk_preloadE'),
]:
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, sym.encode())
    kernels[name] = func

HEADS = 40

def bench(func, seq, iters=10):
    S = torch.zeros(64, 16, dtype=torch.float32, device='cuda')
    Q = torch.randn(HEADS, seq, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(HEADS, seq, 128, device='cuda').to(torch.float8_e4m3fn)
    num_blocks = HEADS * (seq // 32)
    
    args = [ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
            ctypes.c_void_p(K.data_ptr()), ctypes.c_uint32(seq), ctypes.c_uint32(0)]
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    for _ in range(3):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    end.record()
    end.synchronize()
    
    time_ms = start.elapsed_time(end) / iters
    flops = 2 * HEADS * seq * seq * 128
    return flops / (time_ms * 1e9)

print(f"{'SEQ':>6} | {'SwPipe':>10} | {'Preload':>10} | {'Speedup':>8}")
print("-" * 48)

for seq in [1024, 2048, 4096, 8192, 16384]:
    tf_sw = bench(kernels['SwPipe'], seq)
    tf_pre = bench(kernels['Preload'], seq)
    speedup = tf_pre / tf_sw
    print(f"{seq:>6} | {tf_sw:>8.1f} | {tf_pre:>8.1f} | {speedup:>7.2f}x")
