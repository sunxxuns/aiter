#!/usr/bin/env python3
"""
Debug numerical issues in fwd_fp8_qk_4qtile_v2.s
"""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def test_with_ones():
    """Test with all-ones input (simplest case)."""
    
    co_file = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_4qtile_v2.co"
    func_name = b"_ZN5aiter19fwd_fp8_qk_4qtile_v2E"
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name)
    
    seq_len = 128
    head_dim = 128
    
    # All ones input
    Q = torch.ones(seq_len, head_dim, device='cuda').to(torch.float8_e4m3fn)
    K = torch.ones(seq_len, head_dim, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(seq_len * seq_len, dtype=torch.float32, device='cuda')
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    hip.hipModuleUnload(module)
    
    # Expected: each element = sum over k_tiles of (1*1*128) = 128 * num_k_tiles
    num_k_tiles = (seq_len + 31) // 32
    expected = 128 * num_k_tiles  # = 128 * 4 = 512
    
    kernel_out = O[:128*128].reshape(128, 128)
    
    print("=== All Ones Test ===")
    print(f"Expected value per element: {expected}")
    print(f"Kernel [0,0]: {kernel_out[0,0].item()}")
    print(f"Kernel mean: {kernel_out.mean().item():.2f}")
    print(f"Kernel unique values: {len(torch.unique(kernel_out))}")
    print(f"Kernel histogram:")
    
    # Show value distribution
    flat = kernel_out.flatten()
    for val in torch.unique(flat)[:10]:
        count = (flat == val).sum().item()
        print(f"  {val.item():.1f}: {count} occurrences")

def test_with_preload():
    """Compare 4qtile_v2 with known-working preload kernel."""
    
    # Test with preload kernel (known working)
    co_file = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co"
    func_name = b"_ZN5aiter17fwd_fp8_qk_preloadE"
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name)
    
    seq_len = 64
    head_dim = 128
    
    Q = torch.ones(seq_len, head_dim, device='cuda').to(torch.float8_e4m3fn)
    K = torch.ones(seq_len, head_dim, device='cuda').to(torch.float8_e4m3fn)
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
    
    hip.hipModuleUnload(module)
    
    num_k_tiles = (seq_len + 31) // 32
    expected = 128 * num_k_tiles
    
    print("\n=== Preload Kernel (Reference) ===")
    print(f"Expected value: {expected}")
    print(f"Kernel [0]: {O[0].item()}")
    print(f"Kernel mean (first 1024): {O[:1024].mean().item():.2f}")

def main():
    test_with_ones()
    test_with_preload()
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
If 4qtile_v2 shows incorrect values:
1. Check if it's processing all 4 Q-tiles correctly
2. Check if K-tile iteration is correct
3. Check if output store addresses are correct
4. Check if LDS layout matches expectations
""")

if __name__ == "__main__":
    main()
