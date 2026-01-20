//! Benchmark: Factorial function compiled from C to rv32im to EVM
//!
//! This benchmark measures the EVM gas costs for running factorial(n)
//! where the factorial function was originally written in C and compiled
//! to RISC-V rv32im machine code.
//!
//! C source (conceptual):
//! ```c
//! int factorial(int n) {
//!     int result = 1;
//!     while (n > 0) {
//!         result = result * n;
//!         n = n - 1;
//!     }
//!     return result;
//! }
//! ```
//!
//! Compiled to rv32im assembly:
//! ```asm
//! factorial:
//!     addi t0, zero, 1    # result = 1
//! loop:
//!     beq a0, zero, done  # if n == 0, goto done
//!     mul t0, t0, a0      # result = result * n
//!     addi a0, a0, -1     # n = n - 1
//!     jal zero, -12       # goto loop
//! done:
//!     mv a0, t0           # return result
//!     ecall               # exit
//! ```

use rv32im_evm::{
    compiler::{Compiler, CompilerConfig},
    runtime::Runtime,
};

// Register aliases
const ZERO: u32 = 0;
const T0: u32 = 5;
const A0: u32 = 10;

// Instruction encoding helpers
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

// Instruction helpers
fn addi(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0, rs1, imm) }
fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b000, rs1, rs2, imm) }
fn mul(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000001) }
fn jal(rd: u32, imm: i32) -> u32 { encode_j_type(0b1101111, rd, imm) }
fn add(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000000) }
fn ecall() -> u32 { 0x00000073 }

/// Build the factorial program with input n pre-loaded into a0
fn build_factorial_program(n: u32) -> Vec<u8> {
    // Program structure:
    // 0: addi a0, zero, n     # Load input n into a0
    // 1: addi t0, zero, 1     # result = 1
    // loop:
    // 2: beq a0, zero, 16     # if n == 0, goto done (PC+16 = instruction 6)
    // 3: mul t0, t0, a0       # result = result * n
    // 4: addi a0, a0, -1      # n = n - 1
    // 5: jal zero, -12        # goto loop (PC-12 = instruction 2)
    // done:
    // 6: add a0, t0, zero     # return result (mv a0, t0)
    // 7: ecall                # exit

    let instructions = [
        addi(A0, ZERO, n as i32),  // PC 0: a0 = n
        addi(T0, ZERO, 1),          // PC 4: t0 = 1 (result)
        beq(A0, ZERO, 16),          // PC 8: if a0 == 0, goto PC 24 (done)
        mul(T0, T0, A0),            // PC 12: t0 = t0 * a0
        addi(A0, A0, -1),           // PC 16: a0 = a0 - 1
        jal(ZERO, -12),             // PC 20: goto PC 8 (loop)
        add(A0, T0, ZERO),          // PC 24: a0 = t0 (return result)
        ecall(),                    // PC 28: exit
    ];

    let mut program = Vec::new();
    for instr in &instructions {
        program.extend_from_slice(&instr.to_le_bytes());
    }
    program
}

/// Compile and run factorial(n), returning (result, gas_used)
fn run_factorial(n: u32) -> (u32, u64) {
    let program = build_factorial_program(n);

    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80010000,
        memory_size: 0x20000,
    };

    let mut compiler = Compiler::with_config(config);
    let bytecode = compiler.compile(&program).expect("Compilation failed");

    let runtime = Runtime::new(bytecode);
    let output = runtime.execute().expect("Execution failed");

    (output.return_value, output.gas_used)
}

fn main() {
    println!("=== Factorial Benchmark: C -> rv32im -> EVM ===\n");

    println!("C source code:");
    println!("  int factorial(int n) {{");
    println!("      int result = 1;");
    println!("      while (n > 0) {{");
    println!("          result = result * n;");
    println!("          n = n - 1;");
    println!("      }}");
    println!("      return result;");
    println!("  }}\n");

    println!("Compiled to rv32im ({} instructions, {} bytes):\n", 8, 32);
    println!("  addi a0, zero, n     # Load input");
    println!("  addi t0, zero, 1     # result = 1");
    println!("  loop:");
    println!("  beq  a0, zero, done  # if n == 0, exit loop");
    println!("  mul  t0, t0, a0      # result *= n");
    println!("  addi a0, a0, -1      # n--");
    println!("  jal  zero, loop      # repeat");
    println!("  done:");
    println!("  mv   a0, t0          # return result");
    println!("  ecall                # exit\n");

    println!("Benchmark Results:");
    println!("{:-<60}", "");
    println!("{:^15} | {:^15} | {:^15} | {:^10}", "Input", "Result", "Gas Used", "Gas/Iter");
    println!("{:-<60}", "");

    for n in [3, 5, 7, 10, 12] {
        let (result, gas) = run_factorial(n);
        let expected = (1..=n).product::<u32>();
        assert_eq!(result, expected, "factorial({}) should be {}", n, expected);

        // Estimate gas per loop iteration (subtract base cost, divide by iterations)
        let base_gas = run_factorial(0).1;
        let gas_per_iter = if n > 0 { (gas - base_gas) / n as u64 } else { 0 };

        println!("{:^15} | {:^15} | {:^15} | {:^10}",
            format!("factorial({})", n),
            result,
            gas,
            gas_per_iter
        );
    }
    println!("{:-<60}", "");

    // Show bytecode size
    let program = build_factorial_program(5);
    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80010000,
        memory_size: 0x20000,
    };
    let mut compiler = Compiler::with_config(config);
    let bytecode = compiler.compile(&program).expect("Compilation failed");

    println!("\nCompiled EVM bytecode size: {} bytes", bytecode.len());
    println!("rv32im program size: {} bytes", program.len());
    println!("Expansion ratio: {:.1}x", bytecode.len() as f64 / program.len() as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial_3() {
        let (result, gas) = run_factorial(3);
        assert_eq!(result, 6);
        println!("factorial(3) = {}, gas = {}", result, gas);
    }

    #[test]
    fn test_factorial_5() {
        let (result, gas) = run_factorial(5);
        assert_eq!(result, 120);
        println!("factorial(5) = {}, gas = {}", result, gas);
    }

    #[test]
    fn test_factorial_7() {
        let (result, gas) = run_factorial(7);
        assert_eq!(result, 5040);
        println!("factorial(7) = {}, gas = {}", result, gas);
    }
}
