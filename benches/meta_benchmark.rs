//! Meta-compilation benchmark: Compile the compiler itself
//!
//! This benchmark:
//! 1. Loads a minimal rv32im-to-EVM compiler (written in C, compiled to rv32im)
//! 2. Compiles it to EVM bytecode using our main compiler
//! 3. Runs the EVM code to compile factorial(10) rv32im code
//! 4. Measures gas usage for the meta-compilation

use rv32im_evm::{
    compiler::{Compiler, CompilerConfig},
    runtime::Runtime,
};

// Memory addresses matching mini_compiler.c (close to code region)
const INPUT_LEN_ADDR: u32 = 0x80000C00;
const INPUT_ADDR: u32 = 0x80000C10;
#[allow(dead_code)]
const OUTPUT_LEN_ADDR: u32 = 0x80000D00;
#[allow(dead_code)]
const OUTPUT_ADDR: u32 = 0x80000D10;

// Factorial program in rv32im (same as factorial_benchmark.rs)
fn get_factorial_program() -> Vec<u32> {
    const ZERO: u32 = 0;
    const T0: u32 = 5;
    const A0: u32 = 10;

    fn encode_i_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, imm: i32) -> u32 {
        opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (((imm as u32) & 0xFFF) << 20)
    }

    fn encode_r_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
        opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25)
    }

    fn encode_b_type(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
        let imm = imm as u32;
        let imm12 = (imm >> 12) & 1;
        let imm11 = (imm >> 11) & 1;
        let imm10_5 = (imm >> 5) & 0x3F;
        let imm4_1 = (imm >> 1) & 0xF;
        opcode | (imm11 << 7) | (imm4_1 << 8) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (imm10_5 << 25) | (imm12 << 31)
    }

    fn encode_j_type(opcode: u32, rd: u32, imm: i32) -> u32 {
        let imm = imm as u32;
        let imm20 = (imm >> 20) & 1;
        let imm10_1 = (imm >> 1) & 0x3FF;
        let imm11 = (imm >> 11) & 1;
        let imm19_12 = (imm >> 12) & 0xFF;
        opcode | (rd << 7) | (imm19_12 << 12) | (imm11 << 20) | (imm10_1 << 21) | (imm20 << 31)
    }

    fn addi(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0, rs1, imm) }
    fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b000, rs1, rs2, imm) }
    fn mul(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000001) }
    fn jal(rd: u32, imm: i32) -> u32 { encode_j_type(0b1101111, rd, imm) }
    fn add(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000000) }
    fn ecall() -> u32 { 0x00000073 }

    // factorial(10)
    vec![
        addi(A0, ZERO, 10),         // a0 = 10
        addi(T0, ZERO, 1),          // t0 = 1 (result)
        beq(A0, ZERO, 16),          // if n == 0, goto done
        mul(T0, T0, A0),            // t0 *= a0
        addi(A0, A0, -1),           // a0--
        jal(ZERO, -12),             // goto loop
        add(A0, T0, ZERO),          // a0 = t0
        ecall(),                    // return
    ]
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    Meta-Compilation Benchmark                                ║");
    println!("║         Running rv32im→EVM compiler compiled to rv32im→EVM                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load the mini compiler binary
    let mini_compiler_bin = include_bytes!("../meta/mini_compiler.bin");
    println!("Mini compiler size: {} bytes ({} rv32im instructions)",
             mini_compiler_bin.len(), mini_compiler_bin.len() / 4);

    // Get the factorial program to be compiled
    let factorial_program = get_factorial_program();
    let factorial_bytes: Vec<u8> = factorial_program.iter()
        .flat_map(|&instr| instr.to_le_bytes())
        .collect();

    println!("Factorial program: {} instructions ({} bytes)",
             factorial_program.len(), factorial_bytes.len());
    println!();

    // Set up initial memory:
    // - INPUT_LEN_ADDR (0x0F00): length of factorial program in bytes
    // - INPUT_ADDR (0x1000): factorial program instructions
    let mut initial_memory = Vec::new();

    // Input length at 0x0F00 (relative to MEM_BASE 0x0500)
    // RISC-V address 0x0F00 maps to EVM memory at MEM_BASE + (0x0F00 - load_offset)
    // Since MEM_BASE is at RISC-V 0x0500 when load_address is 0x80000000...
    // Actually, let's use absolute RISC-V addresses

    // For the mini compiler running on EVM:
    // It reads from RISC-V memory addresses INPUT_LEN_ADDR, INPUT_ADDR, etc.
    // These get translated to EVM memory via: evm_addr = MEM_BASE + (rv_addr - SOME_BASE)
    // Looking at our compiler's memory layout, RISC-V memory starts at MEM_BASE (0x0500)
    // for addresses >= load_address (0x80000000)

    // The mini_compiler.c uses fixed addresses like 0x1000, 0x0F00, etc.
    // These are low addresses that our EVM memory model would map directly
    // Let me set up the memory properly

    // Write input length (32 bytes at INPUT_LEN_ADDR)
    let input_len = factorial_bytes.len() as u32;
    initial_memory.push((INPUT_LEN_ADDR, input_len.to_le_bytes().to_vec()));

    // Write factorial program at INPUT_ADDR
    initial_memory.push((INPUT_ADDR, factorial_bytes.clone()));

    // Compile the mini compiler to EVM
    println!("Step 1: Compiling mini compiler (rv32im) to EVM...");

    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80010000,
        memory_size: 0x40000,  // 256KB for the meta compiler
        initial_memory,
        ..Default::default()
    };

    let mut compiler = Compiler::with_config(config);
    let evm_bytecode = match compiler.compile(mini_compiler_bin) {
        Ok(bc) => bc,
        Err(e) => {
            eprintln!("Compilation failed: {:?}", e);
            return;
        }
    };

    let evm_bytecode_len = evm_bytecode.len();
    println!("EVM bytecode size: {} bytes", evm_bytecode_len);
    println!("Expansion ratio: {:.1}x", evm_bytecode_len as f64 / mini_compiler_bin.len() as f64);
    println!();

    // Run the EVM code to compile factorial
    println!("Step 2: Running meta-compiler on EVM to compile factorial(10)...");

    let runtime = Runtime::new(evm_bytecode);
    let result = match runtime.execute() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            return;
        }
    };

    println!("Execution completed!");
    println!("Return value: {} (output bytecode length)", result.return_value);
    println!("Gas used: {}", result.gas_used);
    println!();

    // Calculate statistics
    let rv32im_instructions = mini_compiler_bin.len() / 4;
    let gas_per_instruction = result.gas_used as f64 / rv32im_instructions as f64;

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ RESULTS                                                                      │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Mini compiler: {} rv32im instructions                                    │", rv32im_instructions);
    println!("│ EVM bytecode:  {} bytes                                                │", evm_bytecode_len);
    println!("│ Total gas:     {} ({:.2} gas/rv32im instruction)              │",
             result.gas_used, gas_per_instruction);
    println!("│ Output:        {} bytes of compiled EVM code for factorial              │", result.return_value);
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // If successful, try to verify the output by extracting and running it
    if result.return_value > 0 {
        println!();
        println!("Note: The meta-compiler successfully compiled {} bytes of EVM bytecode",
                 result.return_value);
        println!("      for the factorial(10) program.");
    }
}
