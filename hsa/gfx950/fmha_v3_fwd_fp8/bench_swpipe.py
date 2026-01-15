import torch, ctypes, os, time
os.environ['HIP_VISIBLE_DEVICES'] = '0'
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

# Load both kernels
module_old = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module_old), b'fwd_fp8_qk_pipe.co')
func_old = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func_old), module_old, b'_ZN5aiter14fwd_fp8_qk_pipeE')

module_new = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module_new), b'fwd_fp8_qk_swpipe.co')
func_new = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func_new), module_new, b'_ZN5aiter16fwd_fp8_qk_swpipeE')

HEADS = 40
SEQ = 8192
S = torch.zeros(64, 16, dtype=torch.float32, device='cuda')
Q = torch.randn(HEADS, SEQ, 128, device='cuda').to(torch.float8_e4m3fn)
K = torch.randn(HEADS, SEQ, 128, device='cuda').to(torch.float8_e4m3fn)

num_blocks = HEADS * (SEQ // 32)
print(f"Benchmarking: H={HEADS}, SEQ={SEQ}, blocks={num_blocks}")

def bench_kernel(func, name, iters=20):
    args = [ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
            ctypes.c_void_p(K.data_ptr()), ctypes.c_uint32(SEQ), ctypes.c_uint32(0)]
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Warmup
    for _ in range(5):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    end.record()
    end.synchronize()
    
    time_ms = start.elapsed_time(end) / iters
    flops = 2 * HEADS * SEQ * SEQ * 128
    tflops = flops / (time_ms * 1e9)
    print(f"{name}: {time_ms*1000:.1f} us, {tflops:.1f} TF/s")
    return time_ms

t_old = bench_kernel(func_old, "Original (lgkmcnt(0))")
t_new = bench_kernel(func_new, "SwPipeline (lgkmcnt(2))")
speedup = t_old / t_new
print(f"\nSpeedup: {speedup:.2f}x")
