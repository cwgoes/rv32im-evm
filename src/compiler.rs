//! RISC-V to EVM compiler
//!
//! Compiles RISC-V rv32im instructions into EVM bytecode.
//!
//! # Memory Layout in EVM
//!
//! The EVM memory is laid out as follows:
//! - 0x0000-0x03FF: RISC-V registers (x0-x31, 32 bytes each = 1024 bytes)
//! - 0x0400-0x041F: Program counter (PC)
//! - 0x0420-0x043F: Return value storage
//! - 0x0500+: RISC-V memory
//!
//! All values are stored in big-endian format in the high bits of 32-byte EVM words.
//!
//! # Stack Model and DUP Optimization
//!
//! The compiler maintains a stack model that tracks what register values are
//! currently on the EVM stack. When a register value is needed and it's already
//! on the stack (within DUP1-DUP16 range), we use DUP instead of MLOAD.
//!
//! ## Register Caching (Experimental)
//!
//! The compiler supports optional register caching within basic blocks, controlled
//! by `CompilerConfig::enable_register_caching`. When enabled, register values
//! are DUP'd before MSTORE so they remain on the stack for potential reuse.
//!
//! **Currently disabled by default** because the cleanup overhead at branch
//! points (SWAP + POPs) typically exceeds the savings in typical loop patterns.
//! For a loop with N stores and 1 cache hit before branch:
//! - Extra costs: N×DUP(3) + SWAP(3) + N×POP(2) = 5N+3 gas
//! - Savings: ~9 gas per cache hit
//! - Breakeven: Need 5N+3 < 9×hits, rarely achieved in practice
//!
//! The optimization may benefit longer straight-line code sequences with
//! multiple register reuses, but typical RISC-V loops don't see improvement.

use crate::decoder::{decode_instruction, Instruction};
use crate::evm::{EvmBytecode, Opcode};
use std::collections::HashMap;

/// Stack entry type for tracking what's on the EVM stack
#[derive(Debug, Clone, Copy, PartialEq)]
enum StackEntry {
    /// A RISC-V register value (0-31)
    Register(u8),
    /// Some computed/unknown value
    Unknown,
}

/// Stack model for tracking EVM stack contents
///
/// This enables cross-instruction register caching by tracking what register
/// values are currently on the stack. When a register needs to be loaded,
/// we can use DUP if the value is already on the stack (depth 1-16).
///
/// The model is cleared at basic block boundaries (JUMPDEST) since we cannot
/// know the stack state when jumping from other locations.
#[derive(Debug, Clone)]
struct StackModel {
    /// Stack entries, index 0 = bottom, last = top
    entries: Vec<StackEntry>,
    /// Maximum tracked depth (EVM DUP1-DUP16)
    max_depth: usize,
}

impl StackModel {
    fn new() -> Self {
        Self {
            entries: Vec::with_capacity(32),
            max_depth: 16,
        }
    }

    /// Push a register value onto the stack
    fn push_register(&mut self, reg: u8) {
        self.entries.push(StackEntry::Register(reg));
        self.trim();
    }

    /// Push an unknown value onto the stack
    fn push_unknown(&mut self) {
        self.entries.push(StackEntry::Unknown);
        self.trim();
    }

    /// Pop the top value from the stack
    fn pop(&mut self) -> Option<StackEntry> {
        self.entries.pop()
    }

    /// Pop n values from the stack
    fn pop_n(&mut self, n: usize) {
        for _ in 0..n {
            self.entries.pop();
        }
    }

    /// Binary operation: consumes 2 values, produces 1 unknown
    fn binary_op(&mut self) {
        self.pop_n(2);
        self.push_unknown();
    }

    /// Unary operation: consumes 1 value, produces 1 unknown
    fn unary_op(&mut self) {
        self.pop();
        self.push_unknown();
    }

    /// DUP operation: copies value at depth to top
    /// depth is 1-indexed (DUP1 copies top, DUP2 copies second from top)
    fn dup(&mut self, depth: usize) {
        if depth == 0 || depth > self.entries.len() {
            self.push_unknown();
            return;
        }
        let idx = self.entries.len() - depth;
        let entry = self.entries[idx];
        self.entries.push(entry);
        self.trim();
    }

    /// SWAP operation: exchanges top with element at depth
    /// depth is 1-indexed (SWAP1 exchanges top with second)
    fn swap(&mut self, depth: usize) {
        if depth == 0 || depth >= self.entries.len() {
            return;
        }
        let len = self.entries.len();
        self.entries.swap(len - 1, len - 1 - depth);
    }

    /// Find a register on the stack and return its depth (1 = top)
    /// Returns None if not found or depth > 16
    fn find_register(&self, reg: u8) -> Option<usize> {
        if reg == 0 {
            // x0 is always 0, handled separately
            return None;
        }
        for (i, entry) in self.entries.iter().rev().enumerate() {
            let depth = i + 1;
            if depth > self.max_depth {
                break;
            }
            if *entry == StackEntry::Register(reg) {
                return Some(depth);
            }
        }
        None
    }

    /// Invalidate a specific register (when it's written to memory)
    fn invalidate_register(&mut self, reg: u8) {
        for entry in &mut self.entries {
            if *entry == StackEntry::Register(reg) {
                *entry = StackEntry::Unknown;
            }
        }
    }

    /// Mark the top of stack as a specific register value
    /// Used by loop optimization when we update a hot register on the stack
    fn mark_top_as_register(&mut self, reg: u8) {
        if let Some(top) = self.entries.last_mut() {
            *top = StackEntry::Register(reg);
        }
    }

    /// Clear the entire stack model (at basic block boundaries)
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Trim stack to max tracked depth
    fn trim(&mut self) {
        // Keep slightly more than max_depth for better tracking
        let max_track = self.max_depth + 8;
        if self.entries.len() > max_track {
            let remove = self.entries.len() - max_track;
            self.entries.drain(0..remove);
        }
    }

    /// Get current stack depth
    #[allow(dead_code)]
    fn depth(&self) -> usize {
        self.entries.len()
    }
}

// ============================================================
// Loop Detection and Analysis
// ============================================================

/// Information about a detected loop
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LoopInfo {
    /// PC of the loop header (target of back edge)
    header_pc: u32,
    /// PC of the back edge instruction (branch that jumps backward)
    back_edge_pc: u32,
    /// PCs of all instructions in the loop body (from header to back edge)
    body_pcs: Vec<u32>,
    /// Registers that are read in the loop
    regs_read: Vec<u8>,
    /// Registers that are written in the loop
    regs_written: Vec<u8>,
    /// "Hot" registers - both read and written, candidates for stack allocation
    hot_regs: Vec<u8>,
    /// For while-loops: PC of the conditional exit branch (jumps forward out of loop)
    exit_branch_pc: Option<u32>,
    /// For while-loops: target PC of the exit branch
    exit_target_pc: Option<u32>,
    /// True if this is a while-loop (JAL back edge), false if do-while (conditional back edge)
    is_while_loop: bool,
}

impl LoopInfo {
    /// Analyze register usage for a set of instructions in the loop
    fn analyze_registers(instructions: &[(u32, Instruction)], header_pc: u32, back_edge_pc: u32) -> Self {
        use std::collections::HashSet;

        let mut regs_read: HashSet<u8> = HashSet::new();
        let mut regs_written: HashSet<u8> = HashSet::new();
        let mut body_pcs = Vec::new();

        // Collect instructions in the loop body (from header to back edge inclusive)
        for (pc, instr) in instructions {
            if *pc >= header_pc && *pc <= back_edge_pc {
                body_pcs.push(*pc);

                // Analyze register reads and writes
                let (reads, writes) = Self::get_reg_usage(instr);
                for r in reads {
                    if r != 0 { regs_read.insert(r); }
                }
                for w in writes {
                    if w != 0 { regs_written.insert(w); }
                }
            }
        }

        // Hot registers are those that are both read and written
        // Sort for deterministic ordering (HashSet iteration order is arbitrary)
        let mut hot_regs: Vec<u8> = regs_read.intersection(&regs_written)
            .copied()
            .collect();
        hot_regs.sort();

        LoopInfo {
            header_pc,
            back_edge_pc,
            body_pcs,
            regs_read: regs_read.into_iter().collect(),
            regs_written: regs_written.into_iter().collect(),
            hot_regs,
            exit_branch_pc: None,
            exit_target_pc: None,
            is_while_loop: false,
        }
    }

    /// Get registers read and written by an instruction
    fn get_reg_usage(instr: &Instruction) -> (Vec<u8>, Vec<u8>) {
        match instr {
            // R-type: reads rs1, rs2; writes rd
            Instruction::Add { rd, rs1, rs2 } |
            Instruction::Sub { rd, rs1, rs2 } |
            Instruction::Xor { rd, rs1, rs2 } |
            Instruction::Or { rd, rs1, rs2 } |
            Instruction::And { rd, rs1, rs2 } |
            Instruction::Sll { rd, rs1, rs2 } |
            Instruction::Srl { rd, rs1, rs2 } |
            Instruction::Sra { rd, rs1, rs2 } |
            Instruction::Slt { rd, rs1, rs2 } |
            Instruction::Sltu { rd, rs1, rs2 } |
            Instruction::Mul { rd, rs1, rs2 } |
            Instruction::Mulh { rd, rs1, rs2 } |
            Instruction::Mulhsu { rd, rs1, rs2 } |
            Instruction::Mulhu { rd, rs1, rs2 } |
            Instruction::Div { rd, rs1, rs2 } |
            Instruction::Divu { rd, rs1, rs2 } |
            Instruction::Rem { rd, rs1, rs2 } |
            Instruction::Remu { rd, rs1, rs2 } => {
                (vec![*rs1, *rs2], vec![*rd])
            }

            // I-type arithmetic: reads rs1; writes rd
            Instruction::Addi { rd, rs1, .. } |
            Instruction::Xori { rd, rs1, .. } |
            Instruction::Ori { rd, rs1, .. } |
            Instruction::Andi { rd, rs1, .. } |
            Instruction::Slli { rd, rs1, .. } |
            Instruction::Srli { rd, rs1, .. } |
            Instruction::Srai { rd, rs1, .. } |
            Instruction::Slti { rd, rs1, .. } |
            Instruction::Sltiu { rd, rs1, .. } => {
                (vec![*rs1], vec![*rd])
            }

            // Load instructions: reads rs1 (base); writes rd
            Instruction::Lb { rd, rs1, .. } |
            Instruction::Lh { rd, rs1, .. } |
            Instruction::Lw { rd, rs1, .. } |
            Instruction::Lbu { rd, rs1, .. } |
            Instruction::Lhu { rd, rs1, .. } => {
                (vec![*rs1], vec![*rd])
            }

            // Store instructions: reads rs1 (base), rs2 (value); no write
            Instruction::Sb { rs1, rs2, .. } |
            Instruction::Sh { rs1, rs2, .. } |
            Instruction::Sw { rs1, rs2, .. } => {
                (vec![*rs1, *rs2], vec![])
            }

            // Branch instructions: reads rs1, rs2; no write
            Instruction::Beq { rs1, rs2, .. } |
            Instruction::Bne { rs1, rs2, .. } |
            Instruction::Blt { rs1, rs2, .. } |
            Instruction::Bge { rs1, rs2, .. } |
            Instruction::Bltu { rs1, rs2, .. } |
            Instruction::Bgeu { rs1, rs2, .. } => {
                (vec![*rs1, *rs2], vec![])
            }

            // LUI/AUIPC: no read; writes rd
            Instruction::Lui { rd, .. } |
            Instruction::Auipc { rd, .. } => {
                (vec![], vec![*rd])
            }

            // JAL: no read; writes rd (link register)
            Instruction::Jal { rd, .. } => {
                (vec![], vec![*rd])
            }

            // JALR: reads rs1; writes rd
            Instruction::Jalr { rd, rs1, .. } => {
                (vec![*rs1], vec![*rd])
            }

            // System instructions
            Instruction::Ecall | Instruction::Ebreak | Instruction::Fence { .. } => {
                (vec![], vec![])
            }

            Instruction::Unknown { .. } => (vec![], vec![]),
        }
    }

    /// Check if this loop is suitable for stack-based optimization
    /// Returns true if:
    /// - Loop has 2-6 hot registers (can fit in DUP range)
    /// - Loop is not too large (< 20 instructions)
    /// - Back edge is a conditional branch (so exit is via fallthrough)
    fn is_optimizable(&self) -> bool {
        let hot_count = self.hot_regs.len();
        hot_count >= 1 && hot_count <= 6 && self.body_pcs.len() <= 20
    }

    /// Check if the back edge is a conditional branch (not JAL/J)
    #[allow(dead_code)]
    fn has_conditional_back_edge(instructions: &[(u32, Instruction)], back_edge_pc: u32) -> bool {
        for (pc, instr) in instructions {
            if *pc == back_edge_pc {
                return matches!(instr,
                    Instruction::Beq { .. } |
                    Instruction::Bne { .. } |
                    Instruction::Blt { .. } |
                    Instruction::Bge { .. } |
                    Instruction::Bltu { .. } |
                    Instruction::Bgeu { .. }
                );
            }
        }
        false
    }
}

/// Find the exit branch in a while-loop (conditional forward jump out of the loop)
/// Returns (exit_branch_pc, exit_target_pc) if there is exactly ONE exit branch.
/// Returns (None, None) if there are multiple exits (not optimizable) or no exits.
fn find_exit_branch(instructions: &[(u32, Instruction)], header_pc: u32, back_edge_pc: u32) -> (Option<u32>, Option<u32>) {
    let mut exit_branches = Vec::new();

    // Look for conditional branches within the loop that jump forward past the back edge
    for (pc, instr) in instructions {
        if *pc >= header_pc && *pc < back_edge_pc {
            let target = match instr {
                Instruction::Beq { imm, .. } |
                Instruction::Bne { imm, .. } |
                Instruction::Blt { imm, .. } |
                Instruction::Bge { imm, .. } |
                Instruction::Bltu { imm, .. } |
                Instruction::Bgeu { imm, .. } => {
                    Some(pc.wrapping_add(*imm as u32))
                }
                _ => None,
            };

            if let Some(target_pc) = target {
                // Forward jump past the back edge = loop exit
                if target_pc > back_edge_pc {
                    exit_branches.push((*pc, target_pc));
                }
            }
        }
    }

    // Only optimize if there's exactly one exit branch
    // Multiple exits are too complex to handle with the current approach
    if exit_branches.len() == 1 {
        (Some(exit_branches[0].0), Some(exit_branches[0].1))
    } else {
        (None, None)
    }
}

/// Detect loops in the instruction stream
/// Handles two patterns:
/// 1. Do-while loops: conditional back edge (BEQ/BNE jumping backward)
/// 2. While loops: JAL back edge + conditional forward exit branch
fn detect_loops(instructions: &[(u32, Instruction)], load_address: u32) -> Vec<LoopInfo> {
    let mut loops = Vec::new();

    for (pc, instr) in instructions {
        // Pattern 1: JAL back edge (while-loop)
        // The exit is via a conditional forward branch inside the loop
        if let Instruction::Jal { rd, imm } = instr {
            if *rd == 0 {  // J pseudo-instruction (jal x0, offset)
                let target_pc = pc.wrapping_add(*imm as u32);
                if target_pc < *pc && target_pc >= load_address {
                    // Find the conditional exit branch
                    let (exit_branch_pc, exit_target_pc) = find_exit_branch(instructions, target_pc, *pc);

                    // Only optimize if we found a clear exit branch
                    if exit_branch_pc.is_some() {
                        let mut loop_info = LoopInfo::analyze_registers(instructions, target_pc, *pc);
                        loop_info.exit_branch_pc = exit_branch_pc;
                        loop_info.exit_target_pc = exit_target_pc;
                        loop_info.is_while_loop = true;
                        if loop_info.is_optimizable() {
                            loops.push(loop_info);
                        }
                    }
                }
            }
        }

        // Pattern 2: Conditional back edge (do-while loop)
        // The exit is via fallthrough when branch not taken
        let target = match instr {
            Instruction::Beq { imm, .. } |
            Instruction::Bne { imm, .. } |
            Instruction::Blt { imm, .. } |
            Instruction::Bge { imm, .. } |
            Instruction::Bltu { imm, .. } |
            Instruction::Bgeu { imm, .. } => {
                Some(pc.wrapping_add(*imm as u32))
            }
            _ => None,
        };

        if let Some(target_pc) = target {
            // Backward branch = do-while loop
            if target_pc < *pc && target_pc >= load_address {
                let mut loop_info = LoopInfo::analyze_registers(instructions, target_pc, *pc);
                loop_info.is_while_loop = false;
                if loop_info.is_optimizable() {
                    loops.push(loop_info);
                }
            }
        }
    }

    loops
}

/// Stack layout for loop optimization
/// Maps hot registers to their fixed positions on the EVM stack
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LoopStackLayout {
    /// Register to stack depth mapping (depth 1 = top)
    reg_to_depth: HashMap<u8, usize>,
    /// Total number of registers on stack
    count: usize,
}

impl LoopStackLayout {
    fn new(hot_regs: &[u8]) -> Self {
        let mut reg_to_depth = HashMap::new();
        // Assign depths from 1 (top) upward
        for (i, reg) in hot_regs.iter().enumerate() {
            reg_to_depth.insert(*reg, i + 1);
        }
        LoopStackLayout {
            reg_to_depth,
            count: hot_regs.len(),
        }
    }

    fn get_depth(&self, reg: u8) -> Option<usize> {
        self.reg_to_depth.get(&reg).copied()
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
    /// Enable basic block register caching
    /// When true, caches register values on the EVM stack for potential reuse.
    /// Currently disabled by default as the cleanup overhead at branch points
    /// exceeds the savings in typical loop patterns.
    pub enable_register_caching: bool,
    /// Initial memory contents: (RISC-V address, data bytes)
    /// These are written to memory during initialization before program execution.
    pub initial_memory: Vec<(u32, Vec<u8>)>,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000, // 128KB
            enable_register_caching: false, // Disabled: overhead exceeds savings
            initial_memory: Vec::new(),
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
    /// Stack model for cross-instruction DUP optimization
    stack: StackModel,
    /// Number of cached register values currently on the EVM stack
    /// These need to be cleaned up at basic block boundaries
    cached_count: usize,
    /// Detected loops that can be optimized
    detected_loops: Vec<LoopInfo>,
    /// Currently active loop optimization (if any)
    active_loop: Option<LoopInfo>,
    /// Stack layout for the active loop
    loop_stack_layout: Option<LoopStackLayout>,
    /// For while-loops: placeholder positions that need to jump to exit block
    /// (placeholder_position, actual_exit_target_pc)
    while_loop_exit_jumps: Vec<(usize, u32)>,
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
            stack: StackModel::new(),
            cached_count: 0,
            detected_loops: Vec::new(),
            active_loop: None,
            loop_stack_layout: None,
            while_loop_exit_jumps: Vec::new(),
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

        // Detect loops for optimization
        self.detected_loops = detect_loops(&instructions, self.config.load_address);

        // Emit initialization code
        self.emit_init();

        // Second pass: compile each instruction
        for (pc, instr) in &instructions {
            // Check if we're at a loop header
            let loop_at_header = self.get_loop_at_header(*pc).cloned();

            // Check if we're at a back edge (will be handled after instruction)
            let loop_at_back_edge = self.get_loop_at_back_edge(*pc).cloned();

            // Only emit JUMPDEST for actual jump targets
            let is_jump_target = jump_targets.contains(pc);

            if is_jump_target {
                // Clean up any cached values from previous basic block
                // (this happens BEFORE the JUMPDEST for the fallthrough path)
                self.cleanup_cached_values();

                // If this is a loop header, emit prologue BEFORE the JUMPDEST
                // This way: entry path executes prologue then JUMPDEST
                //           back edge jumps to JUMPDEST directly (regs already on stack)
                if let Some(ref loop_info) = loop_at_header {
                    self.emit_loop_prologue(loop_info);
                }

                // Record the EVM position BEFORE emitting JUMPDEST
                // This ensures jump targets point to the JUMPDEST instruction
                self.pc_to_evm.insert(*pc, self.bytecode.position());
                self.bytecode.emit(Opcode::JumpDest);

                // For loop headers, DON'T clear stack - hot regs are on stack
                if loop_at_header.is_none() {
                    // Clear stack model at basic block boundaries
                    // We can't know stack state when jumping from other locations
                    self.stack.clear();
                }
            } else {
                // Not a jump target - record position normally
                self.pc_to_evm.insert(*pc, self.bytecode.position());
            }

            // Compile the instruction
            // Stack model carries over within basic blocks for cross-instruction caching
            self.compile_instruction(*pc, instr)?;

            #[cfg(test)]
            if self.active_loop.is_some() {
                println!("DEBUG after compile PC {:#x}: stack len={}", *pc, self.stack.entries.len());
            }

            // If this was a back edge instruction, handle loop exit
            if loop_at_back_edge.is_some() && self.active_loop.is_some() {
                let is_while_loop = self.active_loop.as_ref().map_or(false, |l| l.is_while_loop);
                if is_while_loop {
                    // For while-loops: emit exit block (JUMPDEST + epilogue + JUMP to exit)
                    // The exit branch jumps here, then we go to actual exit target
                    self.emit_while_loop_exit_block();
                } else {
                    // For do-while loops: emit epilogue for fallthrough exit path
                    // When the conditional back edge is NOT taken, we fall through here
                    self.emit_loop_epilogue();
                }
            }
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

        // Initialize memory from config
        // Clone to avoid borrow issues
        let initial_memory = self.config.initial_memory.clone();
        for (rv_addr, data) in initial_memory {
            self.emit_memory_init(rv_addr, &data);
        }

        // Jump to the main execution loop
        self.bytecode.jump_to("main_entry");
        self.bytecode.jumpdest("main_entry");
    }

    /// Emit code to initialize a memory region
    /// Writes data bytes to RISC-V memory starting at rv_addr
    fn emit_memory_init(&mut self, rv_addr: u32, data: &[u8]) {
        // Convert RISC-V address to EVM address
        let base_evm_addr = MEM_BASE + rv_addr.wrapping_sub(self.config.load_address);

        // Write 32 bytes at a time (EVM word size)
        let mut offset = 0u32;
        while offset < data.len() as u32 {
            let mut word = [0u8; 32];
            let remaining = (data.len() as u32 - offset) as usize;
            let chunk_size = remaining.min(32);

            // Copy bytes to the beginning of the word (big-endian storage)
            word[..chunk_size].copy_from_slice(&data[offset as usize..offset as usize + chunk_size]);

            // Push the 32-byte value
            self.bytecode.emit(Opcode::Push32);
            self.bytecode.emit_bytes(&word);

            // Push the EVM address
            self.bytecode.push_u32(base_evm_addr + offset);

            // MSTORE
            self.bytecode.emit(Opcode::MStore);

            offset += 32;
        }
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
                self.t_push_u32(*imm as u32);
                self.t_binary_op(Opcode::Add);
                self.t_push_u32(0xFFFFFFFE);
                self.t_binary_op(Opcode::And);
                // Dynamic jump dispatch
                self.emit_dynamic_jump();
            }

            // Branch instructions
            // Load operands first (may benefit from cache), then cleanup, then branch
            Instruction::Beq { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                if *rs2 == 0 {
                    // Optimization: beq rs, zero, target
                    // Branch if rs == 0, use ISZERO to convert to condition
                    self.emit_load_reg(*rs1);
                    self.t_unary_op(Opcode::IsZero);
                } else if *rs1 == 0 {
                    // beq zero, rs, target -> same as beq rs, zero
                    self.emit_load_reg(*rs2);
                    self.t_unary_op(Opcode::IsZero);
                } else {
                    // General case: compare two registers
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Eq);
                }
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Bne { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                if *rs2 == 0 {
                    // Optimization: bne rs, zero, target
                    // JUMPI branches if condition != 0, so we can use rs directly
                    self.emit_load_reg(*rs1);
                    // rs value is already the condition (non-zero = branch)
                } else if *rs1 == 0 {
                    // bne zero, rs, target -> same as bne rs, zero
                    self.emit_load_reg(*rs2);
                } else {
                    // General case: compare two registers
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Eq);
                    self.t_unary_op(Opcode::IsZero); // NOT equal
                }
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Blt { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Signed comparison: rs1 < rs2
                self.emit_signed_lt(*rs1, *rs2);
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Bge { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Signed comparison: rs1 >= rs2 (i.e., NOT rs1 < rs2)
                self.emit_signed_lt(*rs1, *rs2);
                self.t_unary_op(Opcode::IsZero);
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Bltu { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Load rs2 first (bottom), then rs1 (top)
                self.emit_load_reg_pair(*rs2, *rs1);
                // LT: returns 1 if s[0] < s[1], i.e., rs1 < rs2
                self.t_binary_op(Opcode::Lt);
                self.emit_branch_with_cleanup(target, fallthrough);
            }

            Instruction::Bgeu { rs1, rs2, imm } => {
                let target = pc.wrapping_add(*imm as u32);
                let fallthrough = pc.wrapping_add(4);
                // Load rs2 first (bottom), then rs1 (top)
                self.emit_load_reg_pair(*rs2, *rs1);
                // LT: returns 1 if rs1 < rs2
                self.t_binary_op(Opcode::Lt);
                self.t_unary_op(Opcode::IsZero); // NOT less than = >=
                self.emit_branch_with_cleanup(target, fallthrough);
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
                    if *imm != 0 {
                        // Non-zero immediate: add and mask
                        self.t_push_u32(*imm as u32);
                        self.t_binary_op(Opcode::Add);
                        self.emit_mask_to_32bit();
                    }
                    // imm == 0: just a register move, no add needed
                    // (register values are already 32-bit bounded)
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
                    self.t_push_u32(*imm as u32);
                    self.emit_load_reg(*rs1);
                    // LT: returns 1 if s[0] < s[1], i.e., rs1 < imm
                    self.t_binary_op(Opcode::Lt);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Xori { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    if *imm != 0 {
                        self.t_push_u32(*imm as u32);
                        self.t_binary_op(Opcode::Xor);
                    }
                    // No mask needed: XOR of 32-bit values is 32-bit
                    // imm == 0: just a register move
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Ori { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    if *imm != 0 {
                        self.t_push_u32(*imm as u32);
                        self.t_binary_op(Opcode::Or);
                    }
                    // No mask needed: OR of 32-bit values is 32-bit
                    // imm == 0: just a register move
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Andi { rd, rs1, imm } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.t_push_u32(*imm as u32);
                    self.t_binary_op(Opcode::And);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Slli { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    if *shamt != 0 {
                        self.t_push1(*shamt);
                        self.t_binary_op(Opcode::Shl);
                        self.emit_mask_to_32bit();
                    }
                    // shamt == 0: just a register move
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srli { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    if *shamt != 0 {
                        self.t_push1(*shamt);
                        self.t_binary_op(Opcode::Shr);
                    }
                    // shamt == 0: just a register move
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srai { rd, rs1, shamt } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    if *shamt != 0 {
                        // Sign-extend to 256 bits, then arithmetic shift right
                        self.emit_sign_extend_32_to_256();
                        self.t_push1(*shamt);
                        self.t_binary_op(Opcode::Sar);
                        self.emit_mask_to_32bit();
                    }
                    // shamt == 0: just a register move
                    self.emit_store_reg(*rd);
                }
            }

            // Register arithmetic instructions
            Instruction::Add { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Add);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sub { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Load rs2 first (bottom), then rs1 (top)
                    // EVM SUB: s[0] - s[1] = rs1 - rs2
                    self.emit_load_reg_pair(*rs2, *rs1);
                    self.t_binary_op(Opcode::Sub);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sll { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_push1(0x1F);
                    self.t_binary_op(Opcode::And); // Only low 5 bits of rs2
                    self.t_binary_op(Opcode::Shl);
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
                    self.emit_load_reg_pair(*rs2, *rs1);
                    // LT: returns 1 if s[0] < s[1], i.e., rs1 < rs2
                    self.t_binary_op(Opcode::Lt);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Xor { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Xor);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Srl { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_push1(0x1F);
                    self.t_binary_op(Opcode::And);
                    self.t_binary_op(Opcode::Shr);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Sra { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg(*rs1);
                    self.emit_sign_extend_32_to_256();
                    self.emit_load_reg(*rs2);
                    self.t_push1(0x1F);
                    self.t_binary_op(Opcode::And);
                    self.t_binary_op(Opcode::Sar);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Or { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Or);
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::And { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::And);
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
                self.cleanup_cached_values();
                self.bytecode.jump_to("halt");
                self.stack.clear();
            }

            Instruction::Ebreak => {
                // For debugging, treat as halt
                self.cleanup_cached_values();
                self.bytecode.jump_to("halt");
                self.stack.clear();
            }

            // M extension (multiply/divide)
            Instruction::Mul { rd, rs1, rs2 } => {
                if *rd != 0 {
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Mul);
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
                    self.t_binary_op(Opcode::Mul);
                    self.t_push1(32);
                    self.t_binary_op(Opcode::Sar); // Arithmetic shift for signed
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
                    self.t_binary_op(Opcode::Mul);
                    self.t_push1(32);
                    self.t_binary_op(Opcode::Sar);
                    self.emit_mask_to_32bit();
                    self.emit_store_reg(*rd);
                }
            }

            Instruction::Mulhu { rd, rs1, rs2 } => {
                if *rd != 0 {
                    // Unsigned * Unsigned, return high 32 bits
                    self.emit_load_reg_pair(*rs1, *rs2);
                    self.t_binary_op(Opcode::Mul);
                    self.t_push1(32);
                    self.t_binary_op(Opcode::Shr);
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
    // Cache management
    // ============================================================

    /// Clean up all cached register values from the EVM stack
    /// Called at basic block boundaries (before JUMPDEST)
    fn cleanup_cached_values(&mut self) {
        for _ in 0..self.cached_count {
            self.bytecode.emit(Opcode::Pop);
            self.stack.pop();
        }
        self.cached_count = 0;
    }

    // ============================================================
    // Tracked bytecode emission methods
    // ============================================================
    // These methods emit bytecode AND update the stack model

    /// Push 0 onto the stack (tracked)
    fn t_push0(&mut self) {
        self.bytecode.push0();
        self.stack.push_unknown();
    }

    /// Push a 1-byte value onto the stack (tracked)
    fn t_push1(&mut self, value: u8) {
        self.bytecode.push1(value);
        self.stack.push_unknown();
    }

    /// Push a 2-byte value onto the stack (tracked)
    #[allow(dead_code)]
    fn t_push2(&mut self, value: u16) {
        self.bytecode.push2(value);
        self.stack.push_unknown();
    }

    /// Push a 4-byte value onto the stack (tracked)
    fn t_push4(&mut self, value: u32) {
        self.bytecode.push4(value);
        self.stack.push_unknown();
    }

    /// Push a u32 value onto the stack (tracked)
    fn t_push_u32(&mut self, value: u32) {
        self.bytecode.push_u32(value);
        self.stack.push_unknown();
    }

    /// Push a u256 value onto the stack (tracked)
    fn t_push_u256(&mut self, bytes: &[u8; 32]) {
        self.bytecode.push_u256(bytes);
        self.stack.push_unknown();
    }

    /// Emit a binary operation (consumes 2, produces 1 unknown)
    fn t_binary_op(&mut self, opcode: Opcode) {
        self.bytecode.emit(opcode);
        self.stack.binary_op();
    }

    /// Emit a unary operation (consumes 1, produces 1 unknown)
    fn t_unary_op(&mut self, opcode: Opcode) {
        self.bytecode.emit(opcode);
        self.stack.unary_op();
    }

    /// Emit MLOAD (consumes address, produces value)
    fn t_mload(&mut self) {
        self.bytecode.emit(Opcode::MLoad);
        self.stack.pop();
        self.stack.push_unknown();
    }

    /// Emit MSTORE (consumes value and address)
    fn t_mstore(&mut self) {
        self.bytecode.emit(Opcode::MStore);
        self.stack.pop_n(2);
    }

    /// Emit MSTORE8 (consumes value and address)
    fn t_mstore8(&mut self) {
        self.bytecode.emit(Opcode::MStore8);
        self.stack.pop_n(2);
    }

    /// Emit a DUP opcode for the given depth (1-16) and track it
    fn t_dup(&mut self, depth: usize) {
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
        self.stack.dup(depth);
    }

    /// Emit a SWAP opcode for the given depth (1-16) and track it
    fn t_swap(&mut self, depth: usize) {
        let opcode = match depth {
            1 => Opcode::Swap1,
            2 => Opcode::Swap2,
            3 => Opcode::Swap3,
            4 => Opcode::Swap4,
            5 => Opcode::Swap5,
            6 => Opcode::Swap6,
            7 => Opcode::Swap7,
            8 => Opcode::Swap8,
            9 => Opcode::Swap9,
            10 => Opcode::Swap10,
            11 => Opcode::Swap11,
            12 => Opcode::Swap12,
            13 => Opcode::Swap13,
            14 => Opcode::Swap14,
            15 => Opcode::Swap15,
            16 => Opcode::Swap16,
            _ => panic!("Invalid SWAP depth: {}", depth),
        };
        self.bytecode.emit(opcode);
        self.stack.swap(depth);
    }

    /// Emit POP and track it
    fn t_pop(&mut self) {
        self.bytecode.emit(Opcode::Pop);
        self.stack.pop();
    }

    // ============================================================
    // Loop Optimization Helpers
    // ============================================================

    /// Check if the given PC is at the start of an optimized loop
    fn get_loop_at_header(&self, pc: u32) -> Option<&LoopInfo> {
        self.detected_loops.iter().find(|l| l.header_pc == pc)
    }

    /// Check if the given PC is the back edge of an optimized loop
    fn get_loop_at_back_edge(&self, pc: u32) -> Option<&LoopInfo> {
        self.detected_loops.iter().find(|l| l.back_edge_pc == pc)
    }

    /// Check if the given PC is the exit branch of a while-loop
    /// Returns Some((actual_exit_target_pc)) if this is an exit branch
    #[allow(dead_code)]
    fn get_while_loop_exit_info(&self, pc: u32) -> Option<u32> {
        if let Some(ref active) = self.active_loop {
            if active.is_while_loop && active.exit_branch_pc == Some(pc) {
                return active.exit_target_pc;
            }
        }
        None
    }

    /// Check if we're currently in an optimized loop and the register is hot
    #[allow(dead_code)]
    fn is_hot_register(&self, reg: u8) -> bool {
        if let Some(ref layout) = self.loop_stack_layout {
            layout.get_depth(reg).is_some()
        } else {
            false
        }
    }

    /// Restore hot registers to the stack model after internal control flow merge
    /// When in a loop, internal control flow (like div-by-zero checks) clears the stack model.
    /// We need to restore it to reflect the actual EVM stack state (hot regs are still there).
    fn restore_hot_regs_in_stack_model(&mut self) {
        self.stack.clear();
        // If in a loop, restore the hot registers in the correct order
        if let Some(ref active) = self.active_loop {
            // hot_regs are ordered so that when loaded in reverse, the first ends up on top
            // Stack after prologue: [last_hot_reg, ..., first_hot_reg] with first on top
            for reg in active.hot_regs.iter().rev() {
                self.stack.push_register(*reg);
            }
        }
    }

    /// Emit loop prologue: load hot registers onto the EVM stack
    /// Stack layout: [reg_n, ..., reg_2, reg_1] with reg_1 on top
    fn emit_loop_prologue(&mut self, loop_info: &LoopInfo) {
        #[cfg(test)]
        println!("DEBUG prologue: BEFORE, stack len={}", self.stack.entries.len());
        // Load hot registers in reverse order so the first one ends up on top
        for reg in loop_info.hot_regs.iter().rev() {
            self.emit_load_reg_from_memory(*reg);
            #[cfg(test)]
            println!("DEBUG prologue: after loading reg {}, stack len={}", reg, self.stack.entries.len());
        }
        // Set up the stack layout
        self.loop_stack_layout = Some(LoopStackLayout::new(&loop_info.hot_regs));
        self.active_loop = Some(loop_info.clone());
        #[cfg(test)]
        println!("DEBUG prologue: AFTER, stack len={}", self.stack.entries.len());
    }

    /// Emit loop epilogue: store hot registers back to memory and clean up stack
    fn emit_loop_epilogue(&mut self) {
        if let Some(ref layout) = self.loop_stack_layout.clone() {
            // Store each hot register from stack back to memory
            // We need to be careful about stack order
            // Stack: [reg_n, ..., reg_2, reg_1] with reg_1 on top
            // Store from top to bottom

            // Get registers in order (depth 1 first = top of stack)
            let mut regs: Vec<(u8, usize)> = layout.reg_to_depth.iter()
                .map(|(r, d)| (*r, *d))
                .collect();
            regs.sort_by_key(|(_, d)| *d);

            #[cfg(test)]
            println!("DEBUG Epilogue: stack model len={}, regs={:?}", self.stack.entries.len(), regs);

            for (reg, depth) in &regs {
                // The top of stack is the register we want to store
                // Stack: [..., reg_value], PUSH addr -> [..., reg_value, addr]
                // MSTORE expects (value, offset) with offset on top
                // MSTORE: stores reg_value at memory[addr]
                let addr = REG_BASE + (*reg as u32) * REG_SIZE;
                #[cfg(test)]
                println!("DEBUG Epilogue: storing reg {} (depth {}) to addr {}", reg, depth, addr);
                let _ = depth; // Suppress unused warning in release mode
                self.bytecode.push_u32(addr);
                self.bytecode.emit(Opcode::MStore);
            }

            // Clear loop state
            self.loop_stack_layout = None;
            self.active_loop = None;
            self.stack.clear();
        }
    }

    /// Emit while-loop exit block: JUMPDEST + epilogue + JUMP to actual exit
    /// This is called after the back edge for while-loops
    /// The exit branch jumps here, then we flush regs and jump to actual exit target
    fn emit_while_loop_exit_block(&mut self) {
        // Record the exit block's position for resolving exit jumps
        let exit_block_position = self.bytecode.position();

        #[cfg(test)]
        println!("DEBUG emit_while_loop_exit_block: position={}, exit_jumps_count={}",
                 exit_block_position, self.while_loop_exit_jumps.len());

        // Emit JUMPDEST for the exit block
        self.bytecode.emit(Opcode::JumpDest);

        // Get the actual exit target before clearing loop state
        let actual_exit_target = if let Some(ref active) = self.active_loop {
            active.exit_target_pc
        } else {
            None
        };

        // Emit epilogue (same as emit_loop_epilogue, but inline)
        if let Some(ref layout) = self.loop_stack_layout.clone() {
            let mut regs: Vec<(u8, usize)> = layout.reg_to_depth.iter()
                .map(|(r, d)| (*r, *d))
                .collect();
            regs.sort_by_key(|(_, d)| *d);

            for (reg, _depth) in &regs {
                let addr = REG_BASE + (*reg as u32) * REG_SIZE;
                self.bytecode.push_u32(addr);
                self.bytecode.emit(Opcode::MStore);
            }
        }

        // Emit JUMP to actual exit target
        if let Some(target_pc) = actual_exit_target {
            self.bytecode.emit(Opcode::Push2);
            self.pending_jumps.push((self.bytecode.position(), target_pc));
            self.bytecode.emit_bytes(&[0, 0]); // Placeholder
            self.bytecode.emit(Opcode::Jump);
        }

        // Now resolve all the while_loop_exit_jumps to point to exit_block_position
        for (placeholder_pos, _actual_target) in self.while_loop_exit_jumps.drain(..) {
            let target_bytes = (exit_block_position as u16).to_be_bytes();
            #[cfg(test)]
            println!("DEBUG: patching placeholder at {} to point to exit_block {}", placeholder_pos, exit_block_position);
            self.bytecode.patch(placeholder_pos, &target_bytes);
        }

        // Clear loop state
        self.loop_stack_layout = None;
        self.active_loop = None;
        self.stack.clear();
    }

    /// Load a register in loop-optimized mode
    /// For hot registers, we rely on the stack model's find_register
    /// to find the current position (since it may change as we compute)
    fn emit_load_reg_loop_optimized(&mut self, _reg: u8) -> bool {
        // Always return false - let the normal emit_load_reg path handle it
        // The stack model's find_register will correctly find hot registers
        // because we push them during prologue and maintain them on the stack
        false
    }

    /// Store a register value in loop-optimized mode
    /// Updates the value on the stack instead of storing to memory
    /// Stack before: [...hot_regs..., ..., new_value]
    /// Stack after: [...hot_regs_updated..., ...]
    fn emit_store_reg_loop_optimized(&mut self, reg: u8) -> bool {
        if reg == 0 {
            self.t_pop();
            return true;
        }

        // Check if we're in an optimized loop and this is a hot register
        if self.active_loop.is_none() {
            return false;
        }

        // Only optimize hot registers
        let is_hot = if let Some(ref layout) = self.loop_stack_layout {
            layout.get_depth(reg).is_some()
        } else {
            false
        };

        if !is_hot {
            return false;
        }

        // Find where the OLD value of this register is on the stack
        if let Some(old_depth) = self.stack.find_register(reg) {
            // Stack: [..., old_reg@depth, ..., new_value@top]
            // new_value is at depth 1, old_reg is at depth old_depth

            if old_depth > 1 && old_depth <= 16 {
                // First, mark the new value (top of stack) as this register
                self.stack.mark_top_as_register(reg);

                // SWAP to exchange new_reg (top) with old_reg (at old_depth)
                // SWAP(n) swaps top with element at depth n+1
                // To swap depth 1 with depth old_depth, use SWAP(old_depth - 1)
                let swap_depth = old_depth - 1;
                self.t_swap(swap_depth);

                // Now: [..., new_reg, ..., old_reg]
                // POP removes the old_reg (now at top)
                self.t_pop();

                return true;
            }
        }
        false
    }

    // ============================================================
    // Helper methods for register access
    // ============================================================

    /// Load a register value onto the EVM stack
    /// Uses loop-optimized DUP when in an optimized loop, otherwise normal load
    fn emit_load_reg(&mut self, reg: u8) {
        // Check for loop optimization first
        if self.emit_load_reg_loop_optimized(reg) {
            return;
        }

        if reg == 0 {
            // x0 is always 0
            self.t_push0();
        } else if let Some(depth) = self.stack.find_register(reg) {
            // Register is already on the stack - use DUP instead of MLOAD
            self.t_dup(depth);
        } else {
            // Load from memory and track as register value
            self.emit_load_reg_from_memory(reg);
        }
    }

    /// Load a register value from memory (always loads, no cache check)
    ///
    /// Values are stored in the low 32 bits of the EVM word, so MLOAD
    /// returns the value directly without needing to shift.
    fn emit_load_reg_from_memory(&mut self, reg: u8) {
        let addr = REG_BASE + (reg as u32) * REG_SIZE;
        // Use t_push_u32 to properly update stack model
        self.t_push_u32(addr);
        self.bytecode.emit(Opcode::MLoad);
        // Update stack: address popped, value pushed
        self.stack.pop();
        // Value is in low 32 bits, no shift needed
        // Mark the result as the register value
        self.stack.push_register(reg);
    }

    /// Load two registers onto the stack (rs1 first/bottom, rs2 second/top)
    /// Uses DUP optimization when possible
    fn emit_load_reg_pair(&mut self, rs1: u8, rs2: u8) {
        self.emit_load_reg(rs1);
        self.emit_load_reg(rs2);
    }

    /// Store the top of stack value into a register
    ///
    /// Values are stored in the low 32 bits of the EVM word (no shifting needed).
    /// When register caching is enabled, DUPs the value before storing so it
    /// remains on the stack for potential reuse within the same basic block.
    fn emit_store_reg(&mut self, reg: u8) {
        // Check for loop optimization first
        if self.emit_store_reg_loop_optimized(reg) {
            return;
        }

        if reg == 0 {
            // Writing to x0 is a no-op, just pop the value
            self.t_pop();
        } else {
            // Invalidate any old cached copies of this register
            self.stack.invalidate_register(reg);

            if self.config.enable_register_caching {
                // Basic block caching: mark the value as this register and DUP it
                // so a copy remains on the stack for potential reuse
                self.stack.pop();  // Remove the Unknown entry
                self.stack.push_register(reg);  // Mark as Register(reg)

                // DUP the value - keeps the original for reuse
                self.t_dup(1);
                self.cached_count += 1;

                // Store in low bits (no shift needed)
                let addr = REG_BASE + (reg as u32) * REG_SIZE;
                self.t_push_u32(addr);
                self.t_mstore();
                // After MSTORE, the original Register(reg) value remains on stack
            } else {
                // Standard behavior: store in low bits (no shift needed)
                let addr = REG_BASE + (reg as u32) * REG_SIZE;
                self.t_push_u32(addr);
                self.t_mstore();
            }
        }
    }

    /// Store an immediate value into a register (stored in low 32 bits)
    fn emit_store_reg_imm(&mut self, reg: u8, value: u32) {
        // Invalidate any cached copies of this register
        self.stack.invalidate_register(reg);
        if reg != 0 {
            self.t_push_u32(value);
            // Store in low bits (no shift needed)
            let addr = REG_BASE + (reg as u32) * REG_SIZE;
            self.t_push_u32(addr);
            self.t_mstore();
        }
    }

    // ============================================================
    // Memory access helpers
    // ============================================================

    /// Compute RISC-V memory address to EVM address
    /// Stack effect: consumes addr, produces EVM addr
    fn emit_rv_to_evm_addr(&mut self) {
        // EVM addr = (rv_addr - load_address) + MEM_BASE
        // Stack: [rv_addr]
        let load_addr = self.config.load_address;
        self.t_push_u32(load_addr);
        // Stack: [rv_addr, load_address] with load_address on top
        self.t_swap(1);
        // Stack: [load_address, rv_addr] with rv_addr on top
        // EVM SUB: s[0] - s[1] = rv_addr - load_address
        self.t_binary_op(Opcode::Sub);
        self.t_push_u32(MEM_BASE);
        self.t_binary_op(Opcode::Add);
    }

    /// Helper to compute EVM address from base register and offset, leaves address on stack
    fn emit_compute_evm_addr(&mut self, base_reg: u8, offset: i32) {
        self.emit_load_reg(base_reg);
        self.t_push_u32(offset as u32);
        self.t_binary_op(Opcode::Add);
        self.emit_rv_to_evm_addr();
    }

    /// Load a byte (signed) from memory
    fn emit_load_byte_signed(&mut self, base_reg: u8, offset: i32) {
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        self.t_push1(248); // 256 - 8
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        // Sign extend from 8 bits
        self.emit_sign_extend_8_to_32();
    }

    /// Load a byte (unsigned) from memory
    fn emit_load_byte_unsigned(&mut self, base_reg: u8, offset: i32) {
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        self.t_push1(248);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
    }

    /// Load a halfword (signed) from little-endian memory
    /// Optimized: single MLOAD + BYTE extraction instead of 2 separate loads
    fn emit_load_half_signed(&mut self, base_reg: u8, offset: i32) {
        // Compute EVM address once
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        // Stack: [mem_value]

        // Extract byte 0 (low byte)
        self.t_dup(1);
        self.t_push1(0);
        self.bytecode.emit(Opcode::Byte);
        self.stack.binary_op();
        // Stack: [mem_value, byte0]

        // Extract byte 1 (high byte) - use SWAP to consume mem_value
        self.t_swap(1);
        self.t_push1(1);
        self.bytecode.emit(Opcode::Byte);
        self.stack.binary_op();
        // Stack: [byte0, byte1]

        // Combine: high_byte << 8 | low_byte
        self.t_push1(8);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);

        // Sign extend from 16 bits
        self.emit_sign_extend_16_to_32();
    }

    /// Load a halfword (unsigned) from little-endian memory
    /// Optimized: single MLOAD + BYTE extraction instead of 2 separate loads
    fn emit_load_half_unsigned(&mut self, base_reg: u8, offset: i32) {
        // Compute EVM address once
        self.emit_compute_evm_addr(base_reg, offset);
        self.t_mload();
        // Stack: [mem_value]

        // Extract byte 0 (low byte)
        self.t_dup(1);
        self.t_push1(0);
        self.bytecode.emit(Opcode::Byte);
        self.stack.binary_op();
        // Stack: [mem_value, byte0]

        // Extract byte 1 (high byte) - use SWAP to consume mem_value
        self.t_swap(1);
        self.t_push1(1);
        self.bytecode.emit(Opcode::Byte);
        self.stack.binary_op();
        // Stack: [byte0, byte1]

        // Combine: high_byte << 8 | low_byte
        self.t_push1(8);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);
    }

    /// Load a word from little-endian memory
    /// Optimized: single MLOAD + BYTE extraction instead of 4 separate loads
    fn emit_load_word(&mut self, base_reg: u8, offset: i32) {
        // Compute EVM address once
        self.emit_compute_evm_addr(base_reg, offset);
        // Stack: [evm_addr]

        // Load 32 bytes from memory
        self.t_mload();
        // Stack: [mem_value] where byte 0 is at bit position 248-255

        // Extract byte 0 (LSB of little-endian word)
        // BYTE(i, x): s[0]=i, s[1]=x, returns x[i] where i=0 is most significant byte
        self.t_dup(1);  // Keep mem_value for later
        // Stack: [mem_value, mem_value]
        self.t_push1(0);
        // Stack: [mem_value, mem_value, 0]
        self.bytecode.emit(Opcode::Byte);
        self.stack.binary_op();
        // Stack: [mem_value, byte0]

        // Extract byte 1
        self.t_dup(2);  // Copy mem_value (now at depth 2)
        // Stack: [mem_value, byte0, mem_value]
        self.t_push1(1);
        // Stack: [mem_value, byte0, mem_value, 1]
        self.bytecode.emit(Opcode::Byte);
        self.stack.binary_op();
        // Stack: [mem_value, byte0, byte1]
        self.t_push1(8);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);
        // Stack: [mem_value, byte0 | (byte1 << 8)]

        // Extract byte 2
        self.t_dup(2);  // Copy mem_value
        self.t_push1(2);
        self.bytecode.emit(Opcode::Byte);
        self.stack.binary_op();
        self.t_push1(16);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);
        // Stack: [mem_value, result_so_far]

        // Extract byte 3 (MSB) - use SWAP to consume mem_value
        self.t_swap(1);
        // Stack: [result_so_far, mem_value]
        self.t_push1(3);
        // Stack: [result_so_far, mem_value, 3]
        self.bytecode.emit(Opcode::Byte);
        self.stack.binary_op();
        // Stack: [result_so_far, byte3]
        self.t_push1(24);
        self.t_binary_op(Opcode::Shl);
        self.t_binary_op(Opcode::Or);
        // Stack: [little-endian word]
    }

    /// Store a byte to memory
    fn emit_store_byte(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        // Load source byte and mask
        self.emit_load_reg(src_reg);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);

        // Compute address
        self.emit_load_reg(base_reg);
        self.t_push_u32(offset as u32);
        self.t_binary_op(Opcode::Add);
        self.emit_rv_to_evm_addr();

        // MSTORE8
        self.t_mstore8();
    }

    /// Store a halfword to memory (little-endian)
    fn emit_store_half(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        // Store low byte
        self.emit_load_reg(src_reg);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.emit_load_reg(base_reg);
        self.t_push_u32(offset as u32);
        self.t_binary_op(Opcode::Add);
        self.emit_rv_to_evm_addr();
        self.t_mstore8();

        // Store high byte
        self.emit_load_reg(src_reg);
        self.t_push1(8);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.emit_load_reg(base_reg);
        self.t_push_u32(offset.wrapping_add(1) as u32);
        self.t_binary_op(Opcode::Add);
        self.emit_rv_to_evm_addr();
        self.t_mstore8();
    }

    /// Store a word to memory (little-endian)
    /// Optimized: compute EVM base address once and reuse for all 4 bytes
    fn emit_store_word(&mut self, base_reg: u8, src_reg: u8, offset: i32) {
        // Compute EVM base address once
        self.emit_compute_evm_addr(base_reg, offset);
        // Stack: [evm_base_addr]

        // Store byte 0 (LSB) - no shift needed
        self.emit_load_reg(src_reg);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        // Stack: [evm_base_addr, byte0]
        self.t_dup(2);  // Copy evm_base_addr
        // Stack: [evm_base_addr, byte0, evm_base_addr]
        self.t_mstore8();
        // Stack: [evm_base_addr]

        // Store byte 1
        self.emit_load_reg(src_reg);
        self.t_push1(8);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.t_dup(2);
        self.t_push1(1);
        self.t_binary_op(Opcode::Add);
        self.t_mstore8();
        // Stack: [evm_base_addr]

        // Store byte 2
        self.emit_load_reg(src_reg);
        self.t_push1(16);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        self.t_dup(2);
        self.t_push1(2);
        self.t_binary_op(Opcode::Add);
        self.t_mstore8();
        // Stack: [evm_base_addr]

        // Store byte 3 (MSB) - use SWAP to consume evm_base_addr
        self.emit_load_reg(src_reg);
        self.t_push1(24);
        self.t_binary_op(Opcode::Shr);
        self.t_push1(0xFF);
        self.t_binary_op(Opcode::And);
        // Stack: [evm_base_addr, byte3]
        self.t_swap(1);
        // Stack: [byte3, evm_base_addr]
        self.t_push1(3);
        self.t_binary_op(Opcode::Add);
        // Stack: [byte3, evm_base_addr+3]
        self.t_mstore8();
        // Stack: []
    }

    // ============================================================
    // Sign extension helpers
    // ============================================================

    /// Sign-extend from 8 bits to 32 bits
    fn emit_sign_extend_8_to_32(&mut self) {
        // EVM SIGNEXTEND(byte_pos, value) where byte_pos is s[0] (top)
        // Stack: [value] -> [value, 0] -> SIGNEXTEND(0, value)
        self.t_push0();
        self.t_binary_op(Opcode::SignExtend);
        self.emit_mask_to_32bit();
    }

    /// Sign-extend from 16 bits to 32 bits
    fn emit_sign_extend_16_to_32(&mut self) {
        // Stack: [value] -> [value, 1] -> SIGNEXTEND(1, value)
        self.t_push1(1);
        self.t_binary_op(Opcode::SignExtend);
        self.emit_mask_to_32bit();
    }

    /// Sign-extend from 32 bits to 256 bits
    fn emit_sign_extend_32_to_256(&mut self) {
        // Stack: [value] -> [value, 3] -> SIGNEXTEND(3, value)
        self.t_push1(3);
        self.t_binary_op(Opcode::SignExtend);
    }

    /// Mask value to 32 bits
    fn emit_mask_to_32bit(&mut self) {
        self.t_push4(0xFFFFFFFF);
        self.t_binary_op(Opcode::And);
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
        self.t_binary_op(Opcode::Slt);
    }

    /// Signed less-than comparison with immediate
    fn emit_signed_lt_imm(&mut self, rs1: u8, imm: i32) {
        // Push the immediate first (will be at bottom of stack)
        if imm >= 0 {
            self.t_push_u32(imm as u32);
        } else {
            // Negative immediate - need to sign-extend to 256 bits
            let mut bytes = [0xFFu8; 32];
            let imm_bytes = (imm as u32).to_be_bytes();
            bytes[28] = imm_bytes[0];
            bytes[29] = imm_bytes[1];
            bytes[30] = imm_bytes[2];
            bytes[31] = imm_bytes[3];
            self.t_push_u256(&bytes);
        }
        // Load rs1 second (will be on top of stack)
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        // SLT: returns 1 if s[0] < s[1], i.e., rs1 < imm
        self.t_binary_op(Opcode::Slt);
    }

    // ============================================================
    // Division helpers (handle division by zero)
    // ============================================================

    /// Signed division
    fn emit_signed_div(&mut self, rs1: u8, rs2: u8) {
        // Clean up cached values - division has complex internal control flow
        self.cleanup_cached_values();

        let div_label = format!("div_{}", self.bytecode.position());
        let end_label = format!("div_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.t_unary_op(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", div_label));
        self.stack.pop(); // JUMPI consumes condition

        // Check for overflow (MIN_INT / -1)
        self.emit_load_reg(rs1);
        self.t_push4(0x80000000); // MIN_INT
        self.t_binary_op(Opcode::Eq);
        self.emit_load_reg(rs2);
        self.t_push4(0xFFFFFFFF); // -1
        self.t_binary_op(Opcode::Eq);
        self.t_binary_op(Opcode::And);
        self.bytecode.jumpi_to(&format!("{}_overflow", div_label));
        self.stack.pop(); // JUMPI consumes condition

        // Normal signed division
        // Load rs2 first (bottom), then rs1 (top)
        // EVM SDIV: s[0] / s[1] = rs1 / rs2
        self.emit_load_reg(rs2);
        self.emit_sign_extend_32_to_256();
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        self.t_binary_op(Opcode::SDiv);
        self.emit_mask_to_32bit();
        self.bytecode.jump_to(&end_label);
        self.restore_hot_regs_in_stack_model();

        // Division by zero: return -1
        self.bytecode.jumpdest(&format!("{}_zero", div_label));
        self.restore_hot_regs_in_stack_model();
        self.t_push4(0xFFFFFFFF);
        self.bytecode.jump_to(&end_label);
        self.restore_hot_regs_in_stack_model();

        // Overflow: return MIN_INT
        self.bytecode.jumpdest(&format!("{}_overflow", div_label));
        self.restore_hot_regs_in_stack_model();
        self.t_push4(0x80000000);

        self.bytecode.jumpdest(&end_label);
        self.restore_hot_regs_in_stack_model();
        self.stack.push_unknown(); // Result is on stack
    }

    /// Unsigned division
    fn emit_unsigned_div(&mut self, rs1: u8, rs2: u8) {
        // Clean up cached values - division has complex internal control flow
        self.cleanup_cached_values();

        let div_label = format!("divu_{}", self.bytecode.position());
        let end_label = format!("divu_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.t_unary_op(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", div_label));
        self.stack.pop(); // JUMPI consumes condition

        // Normal unsigned division
        // Load rs2 first (bottom), then rs1 (top)
        // EVM DIV: s[0] / s[1] = rs1 / rs2
        self.emit_load_reg(rs2);
        self.emit_load_reg(rs1);
        self.t_binary_op(Opcode::Div);
        self.bytecode.jump_to(&end_label);
        self.restore_hot_regs_in_stack_model();

        // Division by zero: return MAX_UINT
        self.bytecode.jumpdest(&format!("{}_zero", div_label));
        self.restore_hot_regs_in_stack_model();
        self.t_push4(0xFFFFFFFF);

        self.bytecode.jumpdest(&end_label);
        self.restore_hot_regs_in_stack_model();
        self.stack.push_unknown(); // Result is on stack
    }

    /// Signed remainder
    fn emit_signed_rem(&mut self, rs1: u8, rs2: u8) {
        // Clean up cached values - remainder has complex internal control flow
        self.cleanup_cached_values();

        let rem_label = format!("rem_{}", self.bytecode.position());
        let end_label = format!("rem_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.t_unary_op(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", rem_label));
        self.stack.pop(); // JUMPI consumes condition

        // Normal signed remainder
        // Load rs2 first (bottom), then rs1 (top)
        // EVM SMOD: s[0] % s[1] = rs1 % rs2
        self.emit_load_reg(rs2);
        self.emit_sign_extend_32_to_256();
        self.emit_load_reg(rs1);
        self.emit_sign_extend_32_to_256();
        self.t_binary_op(Opcode::SMod);
        self.emit_mask_to_32bit();
        self.bytecode.jump_to(&end_label);
        self.restore_hot_regs_in_stack_model();

        // Division by zero: return rs1
        self.bytecode.jumpdest(&format!("{}_zero", rem_label));
        self.restore_hot_regs_in_stack_model();
        self.emit_load_reg(rs1);

        self.bytecode.jumpdest(&end_label);
        self.restore_hot_regs_in_stack_model();
        self.stack.push_unknown(); // Result is on stack
    }

    /// Unsigned remainder
    fn emit_unsigned_rem(&mut self, rs1: u8, rs2: u8) {
        // Clean up cached values - remainder has complex internal control flow
        self.cleanup_cached_values();

        let rem_label = format!("remu_{}", self.bytecode.position());
        let end_label = format!("remu_end_{}", self.bytecode.position());

        // Check for division by zero
        self.emit_load_reg(rs2);
        self.t_unary_op(Opcode::IsZero);
        self.bytecode.jumpi_to(&format!("{}_zero", rem_label));
        self.stack.pop(); // JUMPI consumes condition

        // Normal unsigned remainder
        // Load rs2 first (bottom), then rs1 (top)
        // EVM MOD: s[0] % s[1] = rs1 % rs2
        self.emit_load_reg(rs2);
        self.emit_load_reg(rs1);
        self.t_binary_op(Opcode::Mod);
        self.bytecode.jump_to(&end_label);
        self.restore_hot_regs_in_stack_model();

        // Division by zero: return rs1
        self.bytecode.jumpdest(&format!("{}_zero", rem_label));
        self.restore_hot_regs_in_stack_model();
        self.emit_load_reg(rs1);

        self.bytecode.jumpdest(&end_label);
        self.restore_hot_regs_in_stack_model();
        self.stack.push_unknown(); // Result is on stack
    }

    // ============================================================
    // Jump helpers
    // ============================================================

    /// Emit a jump to a RISC-V PC address
    fn emit_jump_to_rv_pc(&mut self, target_pc: u32) {
        // Clean up cached values before jumping to another basic block
        self.cleanup_cached_values();
        // Record placeholder for later resolution
        self.bytecode.emit(Opcode::Push2);
        self.stack.push_unknown();
        self.pending_jumps.push((self.bytecode.position(), target_pc));
        self.bytecode.emit_bytes(&[0, 0]); // Placeholder
        self.bytecode.emit(Opcode::Jump);
        self.stack.pop(); // JUMP consumes destination
        self.stack.clear(); // Control flow transfer - stack state unknown at target
    }

    /// Emit a conditional branch with proper cleanup
    /// Stack has condition on top, possibly with cached values below
    fn emit_branch_with_cleanup(&mut self, target_pc: u32, _fallthrough_pc: u32) {
        // Stack: [...cached..., condition]
        // Need to cleanup cached values while preserving condition

        if self.cached_count > 0 && self.cached_count <= 16 {
            // Swap condition below cached values
            // Stack: [...cached..., condition] -> [condition, ...cached...]
            self.t_swap(self.cached_count);

            // Pop cached values
            for _ in 0..self.cached_count {
                self.bytecode.emit(Opcode::Pop);
                self.stack.pop();
            }
            self.cached_count = 0;
            // Stack: [condition]
        } else if self.cached_count > 16 {
            // Too many cached values, just clear them all
            // This is rare and not worth optimizing
            for _ in 0..self.cached_count {
                self.bytecode.emit(Opcode::Pop);
                self.stack.pop();
            }
            self.cached_count = 0;
        }

        // Check if this is a while-loop exit branch
        // If so, we need to redirect to the exit block (which will be emitted later)
        let is_while_loop_exit = if let Some(ref active) = self.active_loop {
            active.is_while_loop && active.exit_target_pc == Some(target_pc)
        } else {
            false
        };

        #[cfg(test)]
        if is_while_loop_exit {
            println!("DEBUG: while-loop exit branch detected, target_pc={:#x}", target_pc);
        }

        // JUMPI: s[0] = destination, s[1] = condition
        // Push target addr, then we have [condition, target] with target on top
        self.bytecode.emit(Opcode::Push2);
        self.stack.push_unknown();

        if is_while_loop_exit {
            // For while-loop exit, record in while_loop_exit_jumps
            // The actual target will be the exit block (emitted later)
            #[cfg(test)]
            println!("DEBUG: adding to while_loop_exit_jumps, placeholder at {}", self.bytecode.position());
            self.while_loop_exit_jumps.push((self.bytecode.position(), target_pc));
        } else {
            // Normal branch - record in pending_jumps
            self.pending_jumps.push((self.bytecode.position(), target_pc));
        }

        self.bytecode.emit_bytes(&[0, 0]); // Placeholder
        // Stack is now [condition, target_addr] - correct for JUMPI
        self.bytecode.emit(Opcode::JumpI);
        self.stack.pop_n(2); // JUMPI consumes condition and destination
        #[cfg(test)]
        println!("DEBUG after branch_with_cleanup: stack len={}", self.stack.entries.len());
        // Note: Stack model now has hot registers remaining (if in loop)
    }

    /// Emit dynamic jump based on computed address (for JALR)
    fn emit_dynamic_jump(&mut self) {
        // Stack has target RISC-V PC on top
        // We need to find the corresponding EVM address
        // This requires a dispatch table

        // For simplicity, we'll emit a series of comparisons
        // A more efficient approach would use a jump table

        // Store the target PC temporarily
        self.t_push_u32(PC_ADDR);
        self.t_mstore();

        // Clean up cached values before dynamic jump
        self.cleanup_cached_values();

        // Jump to dispatch routine
        self.bytecode.jump_to("dynamic_dispatch");
        self.stack.clear(); // Dynamic dispatch - stack state unknown
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

#[cfg(test)]
mod loop_optimization_tests {
    use super::*;
    use crate::runtime::Runtime;

    fn encode_r_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
        opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25)
    }

    fn encode_i_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, imm: i32) -> u32 {
        opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (((imm as u32) & 0xFFF) << 20)
    }

    fn encode_b_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
        let imm = imm as u32;
        let imm12 = (imm >> 12) & 1;
        let imm10_5 = (imm >> 5) & 0x3F;
        let imm4_1 = (imm >> 1) & 0xF;
        let imm11 = (imm >> 11) & 1;
        opcode | (imm11 << 7) | (imm4_1 << 8) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (imm10_5 << 25) | (imm12 << 31)
    }

    fn addi(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0b000, rs1, imm) }
    fn add(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000000) }
    fn bne(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b001, rs1, rs2, imm) }
    fn ecall() -> u32 { 0x00000073 }

    const ZERO: u32 = 0;
    const A0: u32 = 10;
    const T0: u32 = 5;

    /// Test: sum 1 to n using a do-while loop with conditional back edge
    /// This pattern CAN be optimized because the back edge is BNE
    ///
    /// loop:
    ///   sum += n
    ///   n--
    ///   bne n, zero, loop  ; conditional back edge
    #[test]
    fn test_dowhile_loop_with_conditional_back_edge() {
        // Sum 1 to 10 = 55
        let instructions = [
            addi(A0, ZERO, 10),         // 0: a0 = n = 10
            addi(T0, ZERO, 0),          // 4: t0 = sum = 0
            // loop header at PC 8:
            add(T0, T0, A0),            // 8: sum += n
            addi(A0, A0, -1),           // 12: n--
            bne(A0, ZERO, -8),          // 16: if n != 0, goto loop (PC 8)
            add(A0, T0, ZERO),          // 20: a0 = sum
            ecall(),                    // 24: return
        ];

        let program: Vec<u8> = instructions.iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();

        let config = CompilerConfig {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000,
            ..Default::default()
        };

        let mut compiler = Compiler::with_config(config);
        let bytecode = compiler.compile(&program).expect("Compilation failed");

        // Debug output
        println!("Detected loops: {}", compiler.detected_loops.len());
        for (i, loop_info) in compiler.detected_loops.iter().enumerate() {
            println!("Loop {}: header={:#x}, back_edge={:#x}, hot_regs={:?}",
                i, loop_info.header_pc, loop_info.back_edge_pc, loop_info.hot_regs);
        }
        println!("Bytecode size: {} bytes", bytecode.len());

        // Compile without optimization for comparison
        let config_no_opt = CompilerConfig {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000,
            ..Default::default()
        };
        let program2: Vec<u8> = [
            addi(A0, ZERO, 10),
            addi(T0, ZERO, 0),
            add(T0, T0, A0),            // Use unconditional back edge
            addi(A0, A0, -1),
            // Can't easily disable opt, just compare sizes
        ].iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut compiler2 = Compiler::with_config(config_no_opt);
        let _ = compiler2.compile(&program2);
        println!("Non-loop bytecode size: {} bytes (partial)", compiler2.bytecode.bytecode().len());

        // Run and verify correctness
        let runtime = Runtime::new(bytecode);
        let output = runtime.execute().expect("Execution failed");
        println!("Result: {}, Gas: {}", output.return_value, output.gas_used);

        // Check that a loop was detected
        assert!(!compiler.detected_loops.is_empty(), "Should detect the do-while loop");
        assert_eq!(output.return_value, 55, "Sum 1 to 10 should be 55");
    }

    /// Compare gas usage: do-while (conditional back edge) vs while (unconditional back edge)
    #[test]
    fn test_compare_loop_styles() {
        // Style 1: Do-while with conditional back edge (can be optimized)
        // Use n=10 to match the failing test
        let dowhile_instructions = [
            addi(A0, ZERO, 10),         // 0: a0 = n = 10
            addi(T0, ZERO, 0),          // 4: t0 = sum = 0
            // loop:
            add(T0, T0, A0),            // 8: sum += n
            addi(A0, A0, -1),           // 12: n--
            bne(A0, ZERO, -8),          // 16: if n != 0, goto loop
            add(A0, T0, ZERO),          // 20: return sum
            ecall(),                    // 24
        ];

        // Style 2: While with unconditional back edge (cannot be optimized)
        fn jal(rd: u32, imm: i32) -> u32 {
            let imm = imm as u32;
            let imm20 = (imm >> 20) & 1;
            let imm10_1 = (imm >> 1) & 0x3FF;
            let imm11 = (imm >> 11) & 1;
            let imm19_12 = (imm >> 12) & 0xFF;
            0b1101111 | (rd << 7) | (imm19_12 << 12) | (imm11 << 20) | (imm10_1 << 21) | (imm20 << 31)
        }
        fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b000, rs1, rs2, imm) }

        let while_instructions = [
            addi(A0, ZERO, 10),         // 0: a0 = n = 10
            addi(T0, ZERO, 0),          // 4: t0 = sum = 0
            // loop:
            beq(A0, ZERO, 16),          // 8: if n == 0, goto done
            add(T0, T0, A0),            // 12: sum += n
            addi(A0, A0, -1),           // 16: n--
            jal(ZERO, -12),             // 20: goto loop (unconditional)
            // done:
            add(A0, T0, ZERO),          // 24: return sum
            ecall(),                    // 28
        ];

        let config = CompilerConfig {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000,
            ..Default::default()
        };

        // Compile and run do-while style
        let dowhile_program: Vec<u8> = dowhile_instructions.iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let mut compiler1 = Compiler::with_config(config.clone());
        let bytecode1 = compiler1.compile(&dowhile_program).expect("Compilation failed");
        let loops_detected = compiler1.detected_loops.len();
        let runtime1 = Runtime::new(bytecode1);
        let output1 = runtime1.execute().expect("Execution failed");

        // Compile and run while style
        let while_program: Vec<u8> = while_instructions.iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let mut compiler2 = Compiler::with_config(config);
        let bytecode2 = compiler2.compile(&while_program).expect("Compilation failed");
        let runtime2 = Runtime::new(bytecode2);
        let output2 = runtime2.execute().expect("Execution failed");

        // Both should produce correct result
        assert_eq!(output1.return_value, 55, "Do-while sum 1-10 should be 55");
        assert_eq!(output2.return_value, 55, "While sum 1-10 should be 55");

        println!("Loop optimization test (sum 1-10):");
        println!("  Do-while (conditional back edge): {} gas, {} loops detected", output1.gas_used, loops_detected);
        println!("  While (unconditional back edge):  {} gas", output2.gas_used);

        if loops_detected > 0 && output1.gas_used < output2.gas_used {
            println!("  Savings: {} gas ({:.1}%)",
                output2.gas_used - output1.gas_used,
                (1.0 - output1.gas_used as f64 / output2.gas_used as f64) * 100.0);
        }
    }

    /// Test: while-loop with 3 hot registers (like GCD)
    #[test]
    fn test_while_loop_3_hot_regs() {
        fn jal(rd: u32, imm: i32) -> u32 {
            let imm = imm as u32;
            let imm20 = (imm >> 20) & 1;
            let imm10_1 = (imm >> 1) & 0x3FF;
            let imm11 = (imm >> 11) & 1;
            let imm19_12 = (imm >> 12) & 0xFF;
            0b1101111 | (rd << 7) | (imm19_12 << 12) | (imm11 << 20) | (imm10_1 << 21) | (imm20 << 31)
        }
        fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b000, rs1, rs2, imm) }
        fn remu(rd: u32, rs1: u32, rs2: u32) -> u32 {
            0b0110011 | (rd << 7) | (0b111 << 12) | (rs1 << 15) | (rs2 << 20) | (0b0000001 << 25)
        }
        fn mv(rd: u32, rs1: u32) -> u32 { add(rd, rs1, ZERO) }
        const T1: u32 = 6;
        const T2: u32 = 7;

        // GCD-like structure: 3 hot registers (T0=5, T1=6, T2=7)
        let instructions = [
            addi(T0, ZERO, 48),         // 0x00: t0 = 48
            addi(T1, ZERO, 18),         // 0x04: t1 = 18
            // loop:
            beq(T1, ZERO, 20),          // 0x08: if t1 == 0, goto done (0x1c)
            remu(T2, T0, T1),           // 0x0c: t2 = t0 % t1
            mv(T0, T1),                 // 0x10: t0 = t1
            mv(T1, T2),                 // 0x14: t1 = t2
            jal(ZERO, -16),             // 0x18: goto loop (0x08)
            // done:
            mv(A0, T0),                 // 0x1c: a0 = t0
            ecall(),                    // 0x20: return
        ];

        let program: Vec<u8> = instructions.iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();

        let config = CompilerConfig {
            load_address: 0x80000000,
            stack_pointer: 0x80010000,
            memory_size: 0x20000,
            ..Default::default()
        };

        let mut compiler = Compiler::with_config(config);
        let bytecode = compiler.compile(&program).expect("Compilation failed");

        // Print detected loops
        println!("Detected loops: {}", compiler.detected_loops.len());
        for (i, loop_info) in compiler.detected_loops.iter().enumerate() {
            println!("Loop {}: header={:#x}, back_edge={:#x}, is_while={}, hot_regs={:?}, exit_pc={:?}, exit_target={:?}",
                i, loop_info.header_pc, loop_info.back_edge_pc, loop_info.is_while_loop,
                loop_info.hot_regs, loop_info.exit_branch_pc, loop_info.exit_target_pc);
        }
        println!("Bytecode size: {} bytes", bytecode.len());

        // Check loop was detected
        assert_eq!(compiler.detected_loops.len(), 1, "Should detect one while-loop");
        let loop_info = &compiler.detected_loops[0];
        assert!(loop_info.is_while_loop, "Should be detected as while-loop");
        assert_eq!(loop_info.hot_regs.len(), 3, "Should have 3 hot registers");

        // Now execute
        let runtime = Runtime::new(bytecode);
        let output = runtime.execute().expect("Execution failed");
        println!("Result: {}, Gas: {}", output.return_value, output.gas_used);
        assert_eq!(output.return_value, 6, "GCD(48, 18) should be 6");
    }
}
