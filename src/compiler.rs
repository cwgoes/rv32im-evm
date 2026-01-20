//! RISC-V to EVM compiler
//!
//! Compiles RISC-V rv32im instructions into EVM bytecode.
//!
//! # Memory Layout in EVM
//!
//! The EVM memory is laid out as follows:
//! - 0x0000-0x007F: RISC-V registers (x0-x31, 4 bytes each = 128 bytes)
//! - 0x0080-0x0083: Program counter (PC)
//! - 0x0084-0x0087: Return value storage
//! - 0x0100+: RISC-V memory (offset by 0x100)
//!
//! All values are stored in little-endian format (RISC-V native).
//!
//! # Stack Caching Optimization
//!
//! The compiler maintains a cache of register values on the EVM stack.
//! When a register value is needed and it's already on the stack, we use
//! DUP instead of MLOAD, saving significant gas (3 gas vs 12+ gas).

use crate::decoder::{decode_instruction, Instruction};
use crate::evm::{EvmBytecode, Opcode};
use std::collections::HashMap;

/// Maximum stack depth for caching (EVM limit is 1024, but we use less for efficiency)
const MAX_CACHE_DEPTH: usize = 16;

/// Entry in the stack cache tracking a register value
#[derive(Debug, Clone, Copy)]
struct CacheEntry {
    /// The RISC-V register number (0-31)
    reg: u8,
    /// Whether this entry is still valid (not overwritten in memory)
    valid: bool,
}

/// Stack cache for register values
///
/// Tracks which register values are currently on the EVM stack.
/// Index 0 is the top of the stack (most recently pushed).
#[derive(Debug, Clone)]
struct StackCache {
    /// Stack entries, index 0 = top of stack
    entries: Vec<CacheEntry>,
}

impl StackCache {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Record that a register value was pushed onto the stack
    fn push(&mut self, reg: u8) {
        self.entries.insert(0, CacheEntry { reg, valid: true });
        // Trim to max depth
        if self.entries.len() > MAX_CACHE_DEPTH {
            self.entries.pop();
        }
    }

    /// Record that the top of stack was popped
    fn pop(&mut self) {
        if !self.entries.is_empty() {
            self.entries.remove(0);
        }
    }

    /// Record that n items were popped from the stack
    fn pop_n(&mut self, n: usize) {
        for _ in 0..n {
            self.pop();
        }
    }

    /// Find a register value in the cache, returns stack depth (1-based for DUP)
    /// Returns None if not found or invalid
    /// NOTE: Cache disabled for now - always returns None
    fn find(&self, _reg: u8) -> Option<usize> {
        // Cache disabled until proper stack tracking is implemented
        None
        /*
        // x0 is always 0, never cached
        if reg == 0 {
            return None;
        }
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.reg == reg && entry.valid {
                // DUP opcodes are 1-indexed (DUP1 = top of stack)
                return Some(i + 1);
            }
        }
        None
        */
    }

    /// Invalidate all cache entries for a register (when it's written to memory)
    fn invalidate(&mut self, reg: u8) {
        for entry in &mut self.entries {
            if entry.reg == reg {
                entry.valid = false;
            }
        }
    }

    /// Clear the entire cache (at control flow boundaries)
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Record that a DUP operation was performed, duplicating item at depth
    fn dup(&mut self, depth: usize) {
        if depth > 0 && depth <= self.entries.len() {
            let entry = self.entries[depth - 1];
            self.entries.insert(0, entry);
            if self.entries.len() > MAX_CACHE_DEPTH {
                self.entries.pop();
            }
        }
    }

    /// Record that a SWAP was performed between top and position depth
    fn swap(&mut self, depth: usize) {
        if depth > 0 && depth <= self.entries.len() {
            self.entries.swap(0, depth);
        }
    }

    /// Push an unknown/non-register value onto the stack
    fn push_unknown(&mut self) {
        // Use reg 255 as a sentinel for non-register values
        self.entries.insert(0, CacheEntry { reg: 255, valid: false });
        if self.entries.len() > MAX_CACHE_DEPTH {
            self.entries.pop();
        }
    }
}

/// Memory layout constants
/// Each register gets a full 32-byte EVM word to avoid overlap issues
pub const REG_BASE: u32 = 0x0000;     // Base address for registers
pub const REG_SIZE: u32 = 32;         // 32 bytes per register (EVM word size)
pub const PC_ADDR: u32 = 0x0400;      // Program counter address (after 32 registers * 32 bytes)
pub const RETVAL_ADDR: u32 = 0x0420;  // Return value storage
pub const MEM_BASE: u32 = 0x0500;     // Base address for RISC-V memory

/// Compiler configuration
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Base address where the RISC-V program is loaded
    pub load_address: u32,
    /// Stack pointer initial value
    pub stack_pointer: u32,
    /// Memory size in bytes
    pub memory_size: u32,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000, // 128KB
        }
    }
}

/// Compiler for RISC-V to EVM translation
pub struct Compiler {
    config: CompilerConfig,
    bytecode: EvmBytecode,
    /// Maps RISC-V PC to EVM bytecode offset
    pc_to_evm: HashMap<u32, usize>,
    /// Maps EVM placeholder positions to RISC-V PC targets
    pending_jumps: Vec<(usize, u32)>,
    /// Stack cache for register value optimization
    stack_cache: StackCache,
}

impl Compiler {
    /// Create a new compiler with default configuration
    pub fn new() -> Self {
        Self::with_config(CompilerConfig::default())
    }

    /// Create a new compiler with custom configuration
    pub fn with_config(config: CompilerConfig) -> Self {
        Self {
            config,
            bytecode: EvmBytecode::new(),
            pc_to_evm: HashMap::new(),
            pending_jumps: Vec::new(),
            stack_cache: StackCache::new(),
        }
    }

    /// Compile a RISC-V program to EVM bytecode
    ///
    /// The input is raw RISC-V machine code bytes.
    /// Returns EVM bytecode that can be executed.
    pub fn compile(&mut self, program: &[u8]) -> Result<Vec<u8>, String> {
        // First pass: decode all instructions and identify jump targets
        let instructions: Vec<(u32, Instruction)> = program
            .chunks_exact(4)
            .enumerate()
            .map(|(i, chunk)| {
                let raw = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let pc = self.config.load_address + (i as u32 * 4);
                (pc, decode_instruction(raw))
            })
            .collect();

        // Identify all jump targets
        let mut jump_targets: std::collections::HashSet<u32> = std::collections::HashSet::new();
        // First instruction is always a target (entry point)
        if let Some((first_pc, _)) = instructions.first() {
            jump_targets.insert(*first_pc);
        }
        // Scan for branch/jump targets
        for (pc, instr) in &instructions {
            match instr {
                Instruction::Jal { imm, .. } => {
                    jump_targets.insert(pc.wrapping_add(*imm as u32));
                }
                Instruction::Beq { imm, .. }
                | Instruction::Bne { imm, .. }
                | Instruction::Blt { imm, .. }
                | Instruction::Bge { imm, .. }
                | Instruction::Bltu { imm, .. }
                | Instruction::Bgeu { imm, .. } => {
                    jump_targets.insert(pc.wrapping_add(*imm as u32));
                }
                Instruction::Jalr { .. } => {
                    // Dynamic jump - all instructions are potential targets
                    // For now, mark all as targets (conservative)
                    for (target_pc, _) in &instructions {
                        jump_targets.insert(*target_pc);
                    }
                }
                _ => {}
            }
        }

        // Emit initialization code
        self.emit_init();

        // Second pass: compile each instruction
        for (pc, instr) in &instructions {
            // Record the EVM position for this RISC-V PC
            self.pc_to_evm.insert(*pc, self.bytecode.position());

            // Only emit JUMPDEST for actual jump targets
            let is_jump_target = jump_targets.contains(pc);
            if is_jump_target {
                self.bytecode.emit(Opcode::JumpDest);
            }

            // Clear cache at start of each instruction
            // This ensures we don't have stale entries from previous instructions
            self.clear_stack_cache();

            // Compile the instruction
            self.compile_instruction(*pc, instr)?;
        }

        // Emit halt/return code
        self.emit_halt();

        // Resolve all pending jumps
        self.resolve_jumps()?;

        // Finalize and return
        self.bytecode.resolve()?;
        Ok(self.bytecode.bytecode().to_vec())
    }

    /// Emit initialization code
    fn emit_init(&mut self) {
        // Initialize x0 (zero) to 0
        self.emit_store_reg_imm(0, 0);

        // Initialize stack pointer (x2/sp)
        self.emit_store_reg_imm(2, self.config.stack_pointer);

        // Jump to the main execution loop
        self.bytecode.jump_to("main_entry");
        self.bytecode.jumpdest("main_entry");
    }

    /// Emit halt code (returns the value in a0/x10)
    fn emit_halt(&mut self) {
        self.bytecode.jumpdest("halt");

        // Load a0 (x10) - the return value
        self.emit_load_reg(10);

        // Store at return position in memory for RETURN
        self.bytecode.push_u32(RETVAL_ADDR);
        self.bytecode.emit(Opcode::MStore);

        // RETURN(offset=RETVAL_ADDR, size=32)
        self.bytecode.push1(32);
        self.bytecode.push_u32(RETVAL_ADDR);
        self.bytecode.emit(Opcode::Return);
    }

    /// Compile a single instruction
    fn compile_instruction(&mut self, pc: u32, instr: &Instruction) -> Result<(), String> {
        match instr {
            // U-type instructions
            Instruction::Lui { rd, imm } => {
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, *imm as u32);
                }
            }

            Instruction::Auipc { rd, imm } => {
                if *rd != 0 {
                    let value = pc.wrapping_add(*imm as u32);
                    self.emit_store_reg_imm(*rd, value);
                }
            }

            // Jump instructions
            Instruction::Jal { rd, imm } => {
                // Store return address (pc + 4) in rd
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, pc.wrapping_add(4));
                }
                // Jump to target
                let target = pc.wrapping_add(*imm as u32);
                self.emit_jump_to_rv_pc(target);
            }

            Instruction::Jalr { rd, rs1, imm } => {
                // Store return address in rd (if rd != 0)
                if *rd != 0 {
                    self.emit_store_reg_imm(*rd, pc.wrapping_add(4));
                }
                // Compute target: (rs1 + imm) & ~1
                self.emit_load_reg(*rs1);
                self.bytecode.push_u32(*imm as u32);
                self.bytecode.emit(Opcode::Add);
                self.bytecode.push_u32(0xFFFFFFFE);
                self.bytecode.emit(Opcode::And);
                // Dynamic jump dispatch
                self.emit_dynamic_jump();
            }

            // Branch instructions
            Instruction::Beq { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                self.emit_load_reg(*rs1);
                self.emit_load_reg(*rs2);
                self.bytecode.emit(Opcode::Eq);
                self.emit_conditional_jump_to_rv_pc(target, fallthrough);
            }

            Instruction::Bne { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                self.emit_load_reg(*rs1);
                self.emit_load_reg(*rs2);
                self.bytecode.emit(Opcode::Eq);
                self.bytecode.emit(Opcode::IsZero); // NOT equal
                self.emit_conditional_jump_to_rv_pc(target, fallthrough);
            }

            Instruction::Blt { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Signed comparison: rs1 < rs2
                self.emit_signed_lt(*rs1, *rs2);
                self.emit_conditional_jump_to_rv_pc(target, fallthrough);
            }

            Instruction::Bge { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Signed comparison: rs1 >= rs2 (i.e., NOT rs1 < rs2)
                self.emit_signed_lt(*rs1, *rs2);
                self.bytecode.emit(Opcode::IsZero);
                self.emit_conditional_jump_to_rv_pc(target, fallthrough);
            }

            Instruction::Bltu { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Load rs2 first (bottom), then rs1 (top)
                self.emit_load_reg(*rs2);
                self.emit_load_reg(*rs1);
                // LT: returns 1 if s[0] < s[1], i.e., rs1 < rs2
                self.bytecode.emit(Opcode::Lt);
                self.emit_conditional_jump_to_rv_pc(target, fallthrough);
            }

            Instruction::Bgeu { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Load rs2 first (bottom), then rs1 (top)
                self.emit_load_reg(*rs2);
                self.emit_load_reg(*rs1);
                // LT: returns 1 if rs1 < rs2
                self.bytecode.emit(Opcode::Lt);
                self.bytecode.emit(Opcode::IsZero); // NOT less than = >=
                self.emit_conditional_jump_to_rv_pc(target, fallthrough);
            }

            // Load instructions
            Instruction::Lb { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_byte_signed(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Lh { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_half_signed(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Lw { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_word(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Lbu { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_byte_unsigned(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Lhu { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_half_unsigned(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            // Store instructions
            Instruction::Sb { rs1, rs2, imm } => {
                self.emit_store_byte(*rs1, *rs2, *imm);
            }

            Instruction::Sh { rs1, rs2, imm } => {
                self.emit_store_half(*rs1, *rs2, *imm);
            }

            Instruction::Sw { rs1, rs2, imm } => {
                self.emit_store_word(*rs1, *rs2, *imm);
            }

            // Arithmetic immediate instructions
            Instruction::Addi { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push_u32(*imm as u32);
                    self.bytecode.emit(Opcode::Add);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Slti { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_signed_lt_imm(*rs1, *imm);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sltiu { rd, rs1, imm } => {
                if *rd != 0 {
                    // Push imm first (bottom), then rs1 (top)
                    self.bytecode.push_u32(*imm as u32);
                    self.emit_load_reg(*rs1);
                    // LT: returns 1 if s[0] < s[1], i.e., rs1 < imm
                    self.bytecode.emit(Opcode::Lt);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Xori { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push_u32(*imm as u32);
                    self.bytecode.emit(Opcode::Xor);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Ori { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push_u32(*imm as u32);
                    self.bytecode.emit(Opcode::Or);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Andi { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push_u32(*imm as u32);
                    self.bytecode.emit(Opcode::And);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Slli { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push1(*shamt);
                    self.bytecode.emit(Opcode::Shl);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srli { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.bytecode.push1(*shamt);
                    self.bytecode.emit(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srai { rd, rs1, shamt } => {
                if *rd != 0 {
                    // Sign-extend to 256 bits, then arithmetic shift right
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.bytecode.push1(*shamt);
                    self.bytecode.emit(Opcode::Sar);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            // Register arithmetic instructions
            Instruction::Add { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Add);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sub { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Load rs2 first (bottom), then rs1 (top)
                    // EVM SUB: s[0] - s[1] = rs1 - rs2
                    self.emit_load_reg(*rs2);
                    self.emit_load_reg(*rs1);
                    self.bytecode.emit(Opcode::Sub);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sll { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.push1(0x1F);
                    self.bytecode.emit(Opcode::And); // Only low 5 bits of rs2
                    self.bytecode.emit(Opcode::Shl);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Slt { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_signed_lt(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sltu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Load rs2 first (bottom), then rs1 (top)
                    self.emit_load_reg(*rs2);
                    self.emit_load_reg(*rs1);
                    // LT: returns 1 if s[0] < s[1], i.e., rs1 < rs2
                    self.bytecode.emit(Opcode::Lt);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Xor { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Xor);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srl { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.push1(0x1F);
                    self.bytecode.emit(Opcode::And);
                    self.bytecode.emit(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sra { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.emit_load_reg(*rs2);
                    self.bytecode.push1(0x1F);
                    self.bytecode.emit(Opcode::And);
                    self.bytecode.emit(Opcode::Sar);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Or { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Or);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::And { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::And);
                    self.emit_store_reg(*rd);
                }
            }

            // System instructions
            Instruction::Fence { .. } => {
                // No-op in EVM (single-threaded)
            }

            Instruction::Ecall => {
                // For testing, ECALL triggers program termination
                // Return value is in a0 (x10)
                self.bytecode.jump_to("halt");
            }

            Instruction::Ebreak => {
                // For debugging, treat as halt
                self.bytecode.jump_to("halt");
            }

            // M extension (multiply/divide)
            Instruction::Mul { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Mul);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Mulh { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Signed * Signed, return high 32 bits
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.emit_load_reg(*rs2);
                    self.emit_sign_extend_32_to_256();
                    self.bytecode.emit(Opcode::Mul);
                    self.bytecode.push1(32);
                    self.bytecode.emit(Opcode::Sar); // Arithmetic shift for signed
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Mulhsu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Signed * Unsigned, return high 32 bits
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.emit_load_reg(*rs2);
                    // rs2 is unsigned, no sign extension needed
                    self.bytecode.emit(Opcode::Mul);
                    self.bytecode.push1(32);
                    self.bytecode.emit(Opcode::Sar);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Mulhu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Unsigned * Unsigned, return high 32 bits
                    self.emit_load_reg(*rs1);
                    self.emit_load_reg(*rs2);
                    self.bytecode.emit(Opcode::Mul);
                    self.bytecode.push1(32);
                    self.bytecode.emit(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Div { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Signed division
                    self.emit_signed_div(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Divu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Unsigned division
                    self.emit_unsigned_div(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Rem { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Signed remainder
                    self.emit_signed_rem(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Remu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Unsigned remainder
                    self.emit_unsigned_rem(*rs1, *rs2);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Unknown { raw } => {
                return Err(format!("Unknown instruction at PC {:#x}: {:#010x}", pc, raw));
            }
        }

        Ok(())
    }

    // ============================================================
    // Helper methods for register access
    // ============================================================

    /// Load a register value onto the EVM stack
    /// Uses simple stack tracking to use DUP when the same register is needed again
    fn emit_load_reg(&mut self, reg: u8) {
        if reg == 0 {
            // x0 is always 0
            self.bytecode.push0();
            self.stack_cache.push(0);
        } else if let Some(depth) = self.stack_cache.find(reg) {
            // Register value is already on the stack - use DUP
            // This happens when the same register is loaded multiple times
            if depth <= 16 {
                self.emit_dup(depth);
                self.stack_cache.dup(depth);
            } else {
                // Too deep for DUP, fall back to memory load
                self.emit_load_reg_from_memory(reg);
            }
        } else {
            // Load from memory
            self.emit_load_reg_from_memory(reg);
        }
    }

    /// Load a register value from memory (no cache check)
    fn emit_load_reg_from_memory(&mut self, reg: u8) {
        let addr = REG_BASE + (reg as u32) * REG_SIZE;
        self.bytecode.push_u32(addr);
        self.bytecode.emit(Opcode::MLoad);
        // Value is in the high bits, shift right to get it
        self.bytecode.push1(224); // 256 - 32 = 224
        self.bytecode.emit(Opcode::Shr);
        self.stack_cache.push(reg);
    }

    /// Emit a DUP opcode for the given depth (1-16)
    fn emit_dup(&mut self, depth: usize) {
        let opcode = match depth {
            1 => Opcode::Dup1,
            2 => Opcode::Dup2,
            3 => Opcode::Dup3,
            4 => Opcode::Dup4,
            5 => Opcode::Dup5,
            6 => Opcode::Dup6,
            7 => Opcode::Dup7,
            8 => Opcode::Dup8,
            9 => Opcode::Dup9,
            10 => Opcode::Dup10,
            11 => Opcode::Dup11,
            12 => Opcode::Dup12,
            13 => Opcode::Dup13,
            14 => Opcode::Dup14,
            15 => Opcode::Dup15,
            16 => Opcode::Dup16,
            _ => panic!("Invalid DUP depth: {}", depth),
        };
        self.bytecode.emit(opcode);
    }

    /// Store the top of stack value into a register
    /// Updates the stack cache to invalidate old values and track the store
    fn emit_store_reg(&mut self, reg: u8) {
        if reg == 0 {
            // Writing to x0 is a no-op, just pop the value
            self.bytecode.emit(Opcode::Pop);
            self.stack_cache.pop();
        } else {
            // Invalidate any cached copies of this register (they're now stale)
            self.stack_cache.invalidate(reg);
            // Shift left to put in high bits, then store
            self.bytecode.push1(224);
            self.bytecode.emit(Opcode::Shl);
            let addr = REG_BASE + (reg as u32) * REG_SIZE;
            self.bytecode.push_u32(addr);
            self.bytecode.emit(Opcode::MStore);
            // The value was consumed from the stack
            self.stack_cache.pop();
        }
    }

    /// Store an immediate value into a register
    fn emit_store_reg_imm(&mut self, reg: u8, value: u32) {
        if reg != 0 {
            // Invalidate any cached copies of this register
            self.stack_cache.invalidate(reg);
            self.bytecode.push_u32(value);
            self.bytecode.push1(224);
            self.bytecode.emit(Opcode::Shl);
            let addr = REG_BASE + (reg as u32) * REG_SIZE;
            self.bytecode.push_u32(addr);
            self.bytecode.emit(Opcode::MStore);
        }
    }

    /// Clear the stack cache (at control flow boundaries)
    fn clear_stack_cache(&mut self) {
        self.stack_cache.clear();
    }

    // ============================================================
    // Memory access helpers
    // ============================================================

    /// Compute RISC-V memory address to EVM address
    /// Stack effect: consumes addr, produces EVM addr
    fn emit_rv_to_evm_addr(&mut self) {
        // EVM addr = (rv_addr - load_address) + MEM_BASE
        // Stack: [rv_addr]
        // Push load_address, swap so rv_addr is on top, then SUB
        self.bytecode.push_u32(self.config.load_address);
        // Stack: [rv_addr, load_address] with load_address on top
        self.bytecode.emit(Opcode::Swap1);
        // Stack: [load_address, rv_addr] with rv_addr on top
        // EVM SUB: s[0] - s[1] = rv_addr - load_address
        self.bytecode.emit(Opcode::Sub);
        self.bytecode.push_u32(MEM_BASE);
        self.bytecode.emit(Opcode::Add);
    }

    /// Helper to compute EVM address from base register and offset, leaves address on stack
    fn emit_compute_evm_addr(&mut self, base_reg: u8, offset: i32) {
        self.emit_load_reg(base_reg);
        self.bytecode.push_u32(offset as u32);
        self.bytecode.emit(Opcode::Add);
        self.emit_rv_to_evm_addr();
    }

    /// Load a byte (signed) from memory
    fn emit_load_byte_signed(&mut self, base_reg: u8, offset: i32) {
        self.emit_compute_evm_addr(base_reg, offset);
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248); // 256 - 8
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);
        // Sign extend from 8 bits
        self.emit_sign_extend_8_to_32();
    }

    /// Load a byte (unsigned) from memory
    fn emit_load_byte_unsigned(&mut self, base_reg: u8, offset: i32) {
        self.emit_compute_evm_addr(base_reg, offset);
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);
    }

    /// Load a halfword (signed) from little-endian memory
    fn emit_load_half_signed(&mut self, base_reg: u8, offset: i32) {
        // Load low byte
        self.emit_compute_evm_addr(base_reg, offset);
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);

        // Load high byte
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(1));
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);

        // Combine: high_byte << 8 | low_byte
        self.bytecode.push1(8);
        self.bytecode.emit(Opcode::Shl);
        self.bytecode.emit(Opcode::Or);

        // Sign extend from 16 bits
        self.emit_sign_extend_16_to_32();
    }

    /// Load a halfword (unsigned) from little-endian memory
    fn emit_load_half_unsigned(&mut self, base_reg: u8, offset: i32) {
        // Load low byte
        self.emit_compute_evm_addr(base_reg, offset);
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);

        // Load high byte
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(1));
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);

        // Combine: high_byte << 8 | low_byte
        self.bytecode.push1(8);
        self.bytecode.emit(Opcode::Shl);
        self.bytecode.emit(Opcode::Or);
    }

    /// Load a word from little-endian memory
    fn emit_load_word(&mut self, base_reg: u8, offset: i32) {
        // Load byte 0 (LSB)
        self.emit_compute_evm_addr(base_reg, offset);
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);

        // Load byte 1
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(1));
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);
        self.bytecode.push1(8);
        self.bytecode.emit(Opcode::Shl);
        self.bytecode.emit(Opcode::Or);

        // Load byte 2
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(2));
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);
        self.bytecode.push1(16);
        self.bytecode.emit(Opcode::Shl);
        self.bytecode.emit(Opcode::Or);

        // Load byte 3 (MSB)
        self.emit_compute_evm_addr(base_reg, offset.wrapping_add(3));
        self.bytecode.emit(Opcode::MLoad);
        self.bytecode.push1(248);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);
        self.bytecode.push1(24);
        self.bytecode.emit(Opcode::Shl);
        self.bytecode.emit(Opcode::Or);
    }

    /// Store a byte to memory
    fn emit_store_byte(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        // Load source byte
        self.emit_load_reg(src_reg);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);

        // Compute address
        self.emit_load_reg(base_reg);
        self.bytecode.push_u32(offset as u32);
        self.bytecode.emit(Opcode::Add);
        self.emit_rv_to_evm_addr();

        // MSTORE8
        self.bytecode.emit(Opcode::MStore8);
    }

    /// Store a halfword to memory (little-endian)
    fn emit_store_half(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        // Store low byte
        self.emit_load_reg(src_reg);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);
        self.emit_load_reg(base_reg);
        self.bytecode.push_u32(offset as u32);
        self.bytecode.emit(Opcode::Add);
        self.emit_rv_to_evm_addr();
        self.bytecode.emit(Opcode::MStore8);

        // Store high byte
        self.emit_load_reg(src_reg);
        self.bytecode.push1(8);
        self.bytecode.emit(Opcode::Shr);
        self.bytecode.push1(0xFF);
        self.bytecode.emit(Opcode::And);
        self.emit_load_reg(base_reg);
        self.bytecode.push_u32(offset.wrapping_add(1) as u32);
        self.bytecode.emit(Opcode::Add);
        self.emit_rv_to_evm_addr();
        self.bytecode.emit(Opcode::MStore8);
    }

    /// Store a word to memory (little-endian)
    fn emit_store_word(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        for i in 0..4 {
            self.emit_load_reg(src_reg);
            self.bytecode.push1((i * 8) as u8);
            self.bytecode.emit(Opcode::Shr);
            self.bytecode.push1(0xFF);
            self.bytecode.emit(Opcode::And);
            self.emit_load_reg(base_reg);
            self.bytecode.push_u32(offset.wrapping_add(i) as u32);
            self.bytecode.emit(Opcode::Add);
            self.emit_rv_to_evm_addr();
            self.bytecode.emit(Opcode::MStore8);
        }
    }

    // ============================================================
    // Sign extension helpers
    // ============================================================

    /// Sign-extend from 8 bits to 32 bits
    fn emit_sign_extend_8_to_32(&mut self) {
        // EVM SIGNEXTEND(byte_pos, value) where byte_pos is s[0] (top)
        // Stack: [value] -> [value, 0] -> SIGNEXTEND(0, value)
        self.bytecode.push0();
        self.bytecode.emit(Opcode::SignExtend);
        self.emit_mask_to_32bit();
    }

    /// Sign-extend from 16 bits to 32 bits
    fn emit_sign_extend_16_to_32(&mut self) {
        // Stack: [value] -> [value, 1] -> SIGNEXTEND(1, value)
        self.bytecode.push1(1);
        self.bytecode.emit(Opcode::SignExtend);
        self.emit_mask_to_32bit();
    }

    /// Sign-extend from 32 bits to 256 bits
    fn emit_sign_extend_32_to_256(&mut self) {
        // Stack: [value] -> [value, 3] -> SIGNEXTEND(3, value)
        self.bytecode.push1(3);
        self.bytecode.emit(Opcode::SignExtend);
    }

    /// Mask value to 32 bits
    /// Note: This modifies the stack so we clear the cache
    fn emit_mask_to_32bit(&mut self) {
        self.bytecode.push4(0xFFFFFFFF);
        self.bytecode.emit(Opcode::And);
        // The top value is now unknown (masked result, not a register)
        // Clear cache to avoid incorrect DUP usage
        self.stack_cache.clear();
    }

    // ============================================================
    // Comparison helpers
    // ============================================================

    /// Signed less-than comparison: rs1 < rs2
    fn emit_signed_lt(&mut self, rs1: u8, rs2: u8) {
        // Load rs2 first (will be at bottom of stack)
        self.emit_load_reg(rs2);
        self.emit_sign_extend_32_to_256();
        // Load rs1 second (will be on top of stack)
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        // SLT: returns 1 if s[0] < s[1], i.e., rs1 < rs2
        self.bytecode.emit(Opcode::Slt);
    }

    /// Signed less-than comparison with immediate
    fn emit_signed_lt_imm(&mut self, rs1: u8, imm: i32) {
        // Push the immediate first (will be at bottom of stack)
        if imm >= 0 {
            self.bytecode.push_u32(imm as u32);
        } else {
            // Negative immediate - need to sign-extend to 256 bits
            let mut bytes = [0xFFu8; 32];
            let imm_bytes = (imm as u32).to_be_bytes();
            bytes[28] = imm_bytes[0];
            bytes[29] = imm_bytes[1];
            bytes[30] = imm_bytes[2];
            bytes[31] = imm_bytes[3];
            self.bytecode.push_u256(&bytes);
        }
        // Load rs1 second (will be on top of stack)
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        // SLT: returns 1 if s[0] < s[1], i.e., rs1 < imm
        self.bytecode.emit(Opcode::Slt);
    }

    // ============================================================
    // Division helpers (handle division by zero)
    // ============================================================

    /// Signed division
    fn emit_signed_div(&mut self, rs1: u8, rs2: u8) {
        let div_label = format!("div_{}", self.bytecode.position());
        let end_label = format!("div_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.bytecode.emit(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", div_label));

        // Check for overflow (MIN_INT / -1)
        self.emit_load_reg(rs1);
        self.bytecode.push4(0x80000000); // MIN_INT
        self.bytecode.emit(Opcode::Eq);
        self.emit_load_reg(rs2);
        self.bytecode.push4(0xFFFFFFFF); // -1
        self.bytecode.emit(Opcode::Eq);
        self.bytecode.emit(Opcode::And);
        self.bytecode.jumpi_to(&format!("{}_overflow", div_label));

        // Normal signed division
        // Load rs2 first (bottom), then rs1 (top)
        // EVM SDIV: s[0] / s[1] = rs1 / rs2
        self.emit_load_reg(rs2);
        self.emit_sign_extend_32_to_256();
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        self.bytecode.emit(Opcode::SDiv);
        self.emit_mask_to_32bit();
        self.bytecode.jump_to(&end_label);

        // Division by zero: return -1
        self.bytecode.jumpdest(&format!("{}_zero", div_label));
        self.bytecode.push4(0xFFFFFFFF);
        self.bytecode.jump_to(&end_label);

        // Overflow: return MIN_INT
        self.bytecode.jumpdest(&format!("{}_overflow", div_label));
        self.bytecode.push4(0x80000000);

        self.bytecode.jumpdest(&end_label);
    }

    /// Unsigned division
    fn emit_unsigned_div(&mut self, rs1: u8, rs2: u8) {
        let div_label = format!("divu_{}", self.bytecode.position());
        let end_label = format!("divu_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.bytecode.emit(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", div_label));

        // Normal unsigned division
        // Load rs2 first (bottom), then rs1 (top)
        // EVM DIV: s[0] / s[1] = rs1 / rs2
        self.emit_load_reg(rs2);
        self.emit_load_reg(rs1);
        self.bytecode.emit(Opcode::Div);
        self.bytecode.jump_to(&end_label);

        // Division by zero: return MAX_UINT
        self.bytecode.jumpdest(&format!("{}_zero", div_label));
        self.bytecode.push4(0xFFFFFFFF);

        self.bytecode.jumpdest(&end_label);
    }

    /// Signed remainder
    fn emit_signed_rem(&mut self, rs1: u8, rs2: u8) {
        let rem_label = format!("rem_{}", self.bytecode.position());
        let end_label = format!("rem_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.bytecode.emit(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", rem_label));

        // Normal signed remainder
        // Load rs2 first (bottom), then rs1 (top)
        // EVM SMOD: s[0] % s[1] = rs1 % rs2
        self.emit_load_reg(rs2);
        self.emit_sign_extend_32_to_256();
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        self.bytecode.emit(Opcode::SMod);
        self.emit_mask_to_32bit();
        self.bytecode.jump_to(&end_label);

        // Division by zero: return rs1
        self.bytecode.jumpdest(&format!("{}_zero", rem_label));
        self.emit_load_reg(rs1);

        self.bytecode.jumpdest(&end_label);
    }

    /// Unsigned remainder
    fn emit_unsigned_rem(&mut self, rs1: u8, rs2: u8) {
        let rem_label = format!("remu_{}", self.bytecode.position());
        let end_label = format!("remu_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.bytecode.emit(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", rem_label));

        // Normal unsigned remainder
        // Load rs2 first (bottom), then rs1 (top)
        // EVM MOD: s[0] % s[1] = rs1 % rs2
        self.emit_load_reg(rs2);
        self.emit_load_reg(rs1);
        self.bytecode.emit(Opcode::Mod);
        self.bytecode.jump_to(&end_label);

        // Division by zero: return rs1
        self.bytecode.jumpdest(&format!("{}_zero", rem_label));
        self.emit_load_reg(rs1);

        self.bytecode.jumpdest(&end_label);
    }

    // ============================================================
    // Jump helpers
    // ============================================================

    /// Emit a jump to a RISC-V PC address
    fn emit_jump_to_rv_pc(&mut self, target_pc: u32) {
        // Record placeholder for later resolution
        self.bytecode.emit(Opcode::Push2);
        self.pending_jumps.push((self.bytecode.position(), target_pc));
        self.bytecode.emit_bytes(&[0, 0]); // Placeholder
        self.bytecode.emit(Opcode::Jump);
    }

    /// Emit a conditional jump to RISC-V PC, with fallthrough
    fn emit_conditional_jump_to_rv_pc(&mut self, target_pc: u32, _fallthrough_pc: u32) {
        // Stack has condition on top
        // JUMPI: s[0] = destination, s[1] = condition
        // Push target addr, then we have [condition, target] with target on top
        self.bytecode.emit(Opcode::Push2);
        self.pending_jumps.push((self.bytecode.position(), target_pc));
        self.bytecode.emit_bytes(&[0, 0]); // Placeholder
        // Stack is now [condition, target_addr] - correct for JUMPI
        self.bytecode.emit(Opcode::JumpI);
        // Fallthrough continues to next instruction
    }

    /// Emit dynamic jump based on computed address (for JALR)
    fn emit_dynamic_jump(&mut self) {
        // Stack has target RISC-V PC on top
        // We need to find the corresponding EVM address
        // This requires a dispatch table

        // For simplicity, we'll emit a series of comparisons
        // A more efficient approach would use a jump table

        // Store the target PC temporarily
        self.bytecode.push_u32(PC_ADDR);
        self.bytecode.emit(Opcode::MStore);

        // Jump to dispatch routine
        self.bytecode.jump_to("dynamic_dispatch");
    }

    /// Resolve all pending jump targets
    fn resolve_jumps(&mut self) -> Result<(), String> {
        for (placeholder_pos, target_pc) in &self.pending_jumps {
            let evm_target = self.pc_to_evm.get(target_pc)
                .ok_or_else(|| format!("Jump target PC {:#x} not found", target_pc))?;
            let target_bytes = (*evm_target as u16).to_be_bytes();
            self.bytecode.bytecode_mut()[*placeholder_pos] = target_bytes[0];
            self.bytecode.bytecode_mut()[*placeholder_pos + 1] = target_bytes[1];
        }
        self.pending_jumps.clear();

        // Now emit the dynamic dispatch routine
        self.emit_dynamic_dispatch();

        Ok(())
    }

    /// Emit the dynamic dispatch routine for JALR
    fn emit_dynamic_dispatch(&mut self) {
        self.bytecode.jumpdest("dynamic_dispatch");

        // Load the target PC from memory
        self.bytecode.push_u32(PC_ADDR);
        self.bytecode.emit(Opcode::MLoad);

        // Generate dispatch table
        for (rv_pc, evm_pos) in self.pc_to_evm.iter() {
            // DUP1 (duplicate target PC)
            self.bytecode.emit(Opcode::Dup1);
            // Push RV PC to compare
            self.bytecode.push_u32(*rv_pc);
            self.bytecode.push1(224);
            self.bytecode.emit(Opcode::Shl);
            // EQ
            self.bytecode.emit(Opcode::Eq);
            // If equal, jump to EVM position
            self.bytecode.push2(*evm_pos as u16);
            self.bytecode.emit(Opcode::Swap1);
            self.bytecode.emit(Opcode::JumpI);
        }

        // If no match found, halt
        self.bytecode.emit(Opcode::Pop); // Pop the target PC
        self.bytecode.jump_to("halt");
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}
