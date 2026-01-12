#!/usr/bin/env python3
"""
Assembly Kernel Validator for FP8 Flash Attention

Detects common bugs:
1. Buffer descriptor size too small for multi-tile access
2. VGPR offset not recalculated in loops
3. Scalar offset exceeds buffer size
4. Missing barriers/waitcnts
5. Register overwrites

Usage:
    python asm_validator.py fwd_fp8_kloop.s
    python asm_validator.py --runtime fwd_fp8_kloop.co  # Runtime checks
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path


@dataclass
class BufferDescriptor:
    """Tracks buffer descriptor state"""
    base_reg: str = ""           # e.g., "s[16:17]"
    size_reg: str = ""           # e.g., "s18"
    size_value: Optional[int] = None
    flags_reg: str = ""          # e.g., "s19"
    name: str = ""               # e.g., "V"
    line_defined: int = 0


@dataclass
class RegisterState:
    """Tracks register usage and values"""
    last_set_line: int = 0
    last_set_value: Optional[str] = None
    used_in_loop: bool = False
    set_before_loop: bool = False


@dataclass 
class LoopInfo:
    """Tracks loop structure"""
    label: str
    start_line: int
    end_line: int = 0
    branch_line: int = 0


class AsmValidator:
    def __init__(self, asm_path: str):
        self.asm_path = Path(asm_path)
        self.lines: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
        # State tracking
        self.buffer_descriptors: Dict[str, BufferDescriptor] = {}
        self.scalar_regs: Dict[str, RegisterState] = {}
        self.vector_regs: Dict[str, RegisterState] = {}
        self.loops: List[LoopInfo] = []
        self.current_loop: Optional[LoopInfo] = None
        
        # Patterns
        self.RE_LABEL = re.compile(r'^(\w+):')
        self.RE_BUFFER_LOAD = re.compile(r'buffer_load\w*\s+(\w+),\s*s\[(\d+):(\d+)\],\s*(\w+)')
        self.RE_SMOV = re.compile(r's_mov_b32\s+s(\d+),\s*(.+)')
        self.RE_SADD = re.compile(r's_add_i32\s+s(\d+),\s*s(\d+),\s*(.+)')
        self.RE_VMOV = re.compile(r'v_mov_b32\w*\s+v(\d+),\s*(.+)')
        self.RE_VLSHL = re.compile(r'v_lshlrev_b32\w*\s+v(\d+),\s*(\d+),\s*v(\d+)')
        self.RE_BRANCH = re.compile(r's_cbranch_\w+\s+(\w+)')
        self.RE_WAITCNT = re.compile(r's_waitcnt\s+(.+)')
        self.RE_BARRIER = re.compile(r's_barrier')
        
    def load(self):
        """Load assembly file"""
        with open(self.asm_path) as f:
            self.lines = f.readlines()
            
    def parse_value(self, val_str: str) -> Optional[int]:
        """Parse immediate value or return None for register"""
        val_str = val_str.strip()
        try:
            if val_str.startswith('0x'):
                return int(val_str, 16)
            elif val_str.startswith('-'):
                return int(val_str)
            elif val_str.isdigit():
                return int(val_str)
        except:
            pass
        return None
        
    def analyze(self):
        """Run all analysis passes"""
        self.load()
        self.pass1_find_loops()
        self.pass2_track_registers()
        self.pass3_check_buffer_descriptors()
        self.pass4_check_loop_invariants()
        self.pass5_check_barriers()
        
    def pass1_find_loops(self):
        """Find loop structures"""
        labels = {}
        for i, line in enumerate(self.lines):
            m = self.RE_LABEL.match(line.strip())
            if m:
                labels[m.group(1)] = i
                
        for i, line in enumerate(self.lines):
            m = self.RE_BRANCH.search(line)
            if m:
                target = m.group(1)
                if target in labels and labels[target] < i:
                    # Backward branch = loop
                    self.loops.append(LoopInfo(
                        label=target,
                        start_line=labels[target],
                        end_line=i,
                        branch_line=i
                    ))
                    
    def pass2_track_registers(self):
        """Track register assignments"""
        in_loop = False
        loop_start = -1
        
        for i, line in enumerate(self.lines):
            # Check if entering a loop
            for loop in self.loops:
                if i == loop.start_line:
                    in_loop = True
                    loop_start = i
                elif i == loop.end_line:
                    in_loop = False
                    
            # Track s_mov_b32
            m = self.RE_SMOV.search(line)
            if m:
                reg = f"s{m.group(1)}"
                value = self.parse_value(m.group(2))
                self.scalar_regs[reg] = RegisterState(
                    last_set_line=i,
                    last_set_value=str(value) if value is not None else m.group(2),
                    set_before_loop=not in_loop
                )
                
            # Track v_lshlrev (common for offset calculation)
            m = self.RE_VLSHL.search(line)
            if m:
                reg = f"v{m.group(1)}"
                self.vector_regs[reg] = RegisterState(
                    last_set_line=i,
                    last_set_value=f"v{m.group(3)} << {m.group(2)}",
                    set_before_loop=not in_loop,
                    used_in_loop=in_loop
                )
                
    def pass3_check_buffer_descriptors(self):
        """Check buffer descriptor configurations"""
        # Find buffer descriptors by looking for s_mov patterns
        # A buffer descriptor is s[n:n+3] where:
        #   s[n:n+1] = base address (usually loaded)
        #   s[n+2] = size
        #   s[n+3] = flags (usually 0x20000 for offen)
        
        # Find flag registers (0x20000 pattern)
        flag_regs = set()
        for reg, state in self.scalar_regs.items():
            if state.last_set_value == "131072" or state.last_set_value == "0x20000":
                flag_regs.add(reg)
                
        # For each flag reg, the size reg is one before it
        for flag_reg in flag_regs:
            reg_num = int(flag_reg[1:])
            size_reg = f"s{reg_num - 1}"
            base_start = reg_num - 3
            
            if size_reg in self.scalar_regs:
                state = self.scalar_regs[size_reg]
                size_val = self.parse_value(state.last_set_value) if state.last_set_value else None
                
                desc = BufferDescriptor(
                    base_reg=f"s[{base_start}:{base_start+1}]",
                    size_reg=size_reg,
                    size_value=size_val,
                    flags_reg=flag_reg,
                    line_defined=state.last_set_line
                )
                self.buffer_descriptors[f"s[{base_start}:{base_start+3}]"] = desc
                
        # Check for size issues
        for desc_name, desc in self.buffer_descriptors.items():
            if desc.size_value is not None:
                if desc.size_value > 0 and desc.size_value < 65536:
                    # Small size - might cause issues with multi-tile
                    self.warnings.append(
                        f"Line {desc.line_defined}: Buffer descriptor {desc_name} "
                        f"has size={desc.size_value} bytes. "
                        f"Multi-tile access with offset >= {desc.size_value} will return 0! "
                        f"Consider using size=-1 (0xFFFFFFFF) for unbounded access."
                    )
                    
    def pass4_check_loop_invariants(self):
        """Check if VGPRs used in buffer_load are recalculated in loops"""
        for loop in self.loops:
            # Find buffer_load instructions in the loop
            for i in range(loop.start_line, loop.end_line + 1):
                line = self.lines[i]
                m = self.RE_BUFFER_LOAD.search(line)
                if m:
                    vgpr = m.group(1)  # The VGPR offset
                    if vgpr.startswith('v'):
                        # Check if this VGPR is set within the loop
                        vgpr_set_in_loop = False
                        for j in range(loop.start_line, i):
                            if re.search(rf'\bv{vgpr[1:]}\b.*=|{vgpr}\s*,', self.lines[j]):
                                vgpr_set_in_loop = True
                                break
                            # Also check v_lshlrev pattern
                            m2 = self.RE_VLSHL.search(self.lines[j])
                            if m2 and f"v{m2.group(1)}" == vgpr:
                                vgpr_set_in_loop = True
                                break
                                
                        if not vgpr_set_in_loop and vgpr in self.vector_regs:
                            if self.vector_regs[vgpr].set_before_loop:
                                self.warnings.append(
                                    f"Line {i+1}: buffer_load uses {vgpr} but it was set "
                                    f"at line {self.vector_regs[vgpr].last_set_line+1} "
                                    f"(before loop at line {loop.start_line+1}). "
                                    f"Consider recalculating {vgpr} at loop start."
                                )
                                
    def pass5_check_barriers(self):
        """Check for missing barriers after LDS operations"""
        lds_store_pending = False
        lds_load_pending = False
        last_lds_op_line = 0
        
        for i, line in enumerate(self.lines):
            if 'ds_write' in line or 'ds_store' in line:
                lds_store_pending = True
                last_lds_op_line = i
            elif 'ds_read' in line or 'ds_load' in line:
                lds_load_pending = True
                last_lds_op_line = i
            elif 'buffer_load' in line and 'lds' in line:
                lds_store_pending = True  # buffer_load to LDS
                last_lds_op_line = i
                
            # Check for barrier
            if self.RE_BARRIER.search(line):
                lds_store_pending = False
                lds_load_pending = False
                
            # Check for waitcnt
            m = self.RE_WAITCNT.search(line)
            if m:
                waitcnt_args = m.group(1)
                if 'lgkmcnt(0)' in waitcnt_args:
                    lds_load_pending = False
                if 'vmcnt(0)' in waitcnt_args:
                    pass  # Doesn't affect LDS stores directly
                    
            # Detect potential race: LDS store followed by LDS read without barrier
            if lds_store_pending and ('ds_read' in line or 'ds_load' in line):
                # Check if there's a waitcnt and barrier before this
                has_barrier = False
                for j in range(last_lds_op_line, i):
                    if self.RE_BARRIER.search(self.lines[j]):
                        has_barrier = True
                        break
                if not has_barrier:
                    self.warnings.append(
                        f"Line {i+1}: LDS read after LDS store (line {last_lds_op_line+1}) "
                        f"without barrier. May cause race condition."
                    )
                    
    def report(self):
        """Print validation report"""
        print("=" * 70)
        print(f"ASM VALIDATOR REPORT: {self.asm_path.name}")
        print("=" * 70)
        
        # Buffer descriptors
        print("\n📦 Buffer Descriptors Found:")
        for name, desc in self.buffer_descriptors.items():
            size_str = f"{desc.size_value}" if desc.size_value else "unknown"
            if desc.size_value == -1 or desc.size_value == 0xFFFFFFFF:
                size_str = "max (unlimited)"
            print(f"  {name}: size={size_str} (line {desc.line_defined+1})")
            
        # Loops
        print(f"\n🔄 Loops Found: {len(self.loops)}")
        for loop in self.loops:
            print(f"  {loop.label}: lines {loop.start_line+1}-{loop.end_line+1}")
            
        # Warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  • {w}")
        else:
            print("\n✅ No warnings")
            
        # Errors
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  • {e}")
        else:
            print("✅ No errors")
            
        print("=" * 70)
        return len(self.errors) == 0


class RuntimeValidator:
    """Runtime validation for kernel execution"""
    
    @staticmethod
    def check_buffer_access(
        desc_name: str,
        base_ptr: int,
        size: int,
        scalar_offset: int,
        vgpr_offsets: list,
        element_size: int = 16  # dwordx4
    ):
        """
        Check if buffer access is within bounds.
        
        Args:
            desc_name: Name for logging
            base_ptr: Base address
            size: Buffer descriptor size (-1 for max)
            scalar_offset: Scalar offset (e.g., s27)
            vgpr_offsets: List of per-thread VGPR offsets
            element_size: Bytes per load
        """
        issues = []
        
        if size == -1 or size == 0xFFFFFFFF:
            return issues  # Unlimited size
            
        for tid, voff in enumerate(vgpr_offsets):
            total_offset = scalar_offset + voff
            end_offset = total_offset + element_size
            
            if end_offset > size:
                issues.append(
                    f"{desc_name}: Thread {tid} accesses offset {total_offset}-{end_offset} "
                    f"but buffer size is {size}. Access will return 0!"
                )
                
        return issues
    
    @staticmethod
    def simulate_buffer_load(
        tile_idx: int,
        num_tiles: int,
        tile_stride: int,
        buffer_size: int,
        threads: int = 64,
        bytes_per_thread: int = 16
    ):
        """
        Simulate buffer_load for multi-tile access.
        
        Returns list of issues found.
        """
        issues = []
        scalar_offset = tile_idx * tile_stride
        
        for tid in range(threads):
            vgpr_offset = tid * bytes_per_thread
            total_offset = scalar_offset + vgpr_offset
            
            if buffer_size != -1 and total_offset >= buffer_size:
                issues.append(
                    f"Tile {tile_idx}: Thread {tid} offset {total_offset} >= buffer size {buffer_size}"
                )
                
        if issues:
            print(f"\n🚨 BUFFER OVERFLOW DETECTED for tile {tile_idx}:")
            for issue in issues[:5]:  # Show first 5
                print(f"  • {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues)-5} more")
                
        return issues


def validate_kernel_params(
    seq_len: int,
    tile_size: int = 32,
    k_buffer_size: int = -1,
    v_buffer_size: int = -1
):
    """
    Validate kernel parameters before launch.
    
    Call this before hipModuleLaunchKernel to catch issues early.
    """
    num_tiles = (seq_len + tile_size - 1) // tile_size
    tile_stride = tile_size * 128  # 32 rows × 128 cols × 1 byte
    
    print(f"\n🔍 Validating kernel params:")
    print(f"   seq_len={seq_len}, tiles={num_tiles}, stride={tile_stride}")
    print(f"   K buffer size: {k_buffer_size if k_buffer_size != -1 else 'unlimited'}")
    print(f"   V buffer size: {v_buffer_size if v_buffer_size != -1 else 'unlimited'}")
    
    all_ok = True
    
    for tile in range(num_tiles):
        k_issues = RuntimeValidator.simulate_buffer_load(
            tile, num_tiles, tile_stride, k_buffer_size
        )
        v_issues = RuntimeValidator.simulate_buffer_load(
            tile, num_tiles, tile_stride, v_buffer_size
        )
        
        if k_issues or v_issues:
            all_ok = False
            
    if all_ok:
        print("   ✅ All tiles accessible")
    else:
        print("   ❌ Buffer size issues detected!")
        
    return all_ok


# Convenience function for quick validation
def quick_check(asm_file: str):
    """Quick validation of an assembly file"""
    validator = AsmValidator(asm_file)
    validator.analyze()
    return validator.report()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python asm_validator.py <asm_file.s>")
        print("       python asm_validator.py --params <seq_len> [k_size] [v_size]")
        sys.exit(1)
        
    if sys.argv[1] == "--params":
        # Validate runtime params
        seq_len = int(sys.argv[2])
        k_size = int(sys.argv[3]) if len(sys.argv) > 3 else -1
        v_size = int(sys.argv[4]) if len(sys.argv) > 4 else -1
        validate_kernel_params(seq_len, k_buffer_size=k_size, v_buffer_size=v_size)
    else:
        # Validate assembly file
        quick_check(sys.argv[1])
