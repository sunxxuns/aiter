import torch, ctypes, os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

# Load kernels
module_old = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module_old), b'fwd_fp8_qk_pipe.co')
func_old = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func_old), module_old, b'_ZN5aiter14fwd_fp8_qk_pipeE')

module_new = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module_new), b'fwd_fp8_qk_swpipe.co')
func_new = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func_new), module_new, b'_ZN5aiter16fwd_fp8_qk_swpipeE')

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
    return flops / (time_ms * 1e9)  # TF/s

print(f"{'SEQ':>6} | {'Original':>12} | {'SwPipeline':>12} | {'Speedup':>8}")
print("-" * 50)

for seq in [1024, 2048, 4096, 8192, 16384]:
    tf_old = bench(func_old, seq)
    tf_new = bench(func_new, seq)
    speedup = tf_new / tf_old
    print(f"{seq:>6} | {tf_old:>10.1f} | {tf_new:>10.1f} | {speedup:>7.2f}x")
