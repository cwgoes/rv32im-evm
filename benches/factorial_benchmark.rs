//! Benchmark: Mathematical functions compiled from C to rv32im to EVM
//!
//! This benchmark measures the EVM gas costs for running various mathematical
//! functions where the code was originally written in C and compiled to RISC-V
//! rv32im machine code, then compiled to EVM bytecode.
//!
//! Gas costs are shown both as total and as "execution gas" (minus the ~21k base).

use rv32im_evm::{
    compiler::{Compiler, CompilerConfig},
    runtime::Runtime,
};

// Register aliases
const ZERO: u32 = 0;
const T0: u32 = 5;
const T1: u32 = 6;
const T2: u32 = 7;
const A0: u32 = 10;
const A1: u32 = 11;

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
fn bge(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b101, rs1, rs2, imm) }
fn bltu(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b110, rs1, rs2, imm) }
fn mul(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000001) }
fn jal(rd: u32, imm: i32) -> u32 { encode_j_type(0b1101111, rd, imm) }
fn add(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000000) }
fn remu(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b111, rs1, rs2, 0b0000001) }
fn ecall() -> u32 { 0x00000073 }

fn compile_and_run(instructions: &[u32]) -> (u32, u64) {
    let mut program = Vec::new();
    for instr in instructions {
        program.extend_from_slice(&instr.to_le_bytes());
    }

    let config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80010000,
        memory_size: 0x20000,
        ..Default::default()
    };

    let mut compiler = Compiler::with_config(config);
    let bytecode = compiler.compile(&program).expect("Compilation failed");

    let runtime = Runtime::new(bytecode);
    let output = runtime.execute().expect("Execution failed");

    (output.return_value, output.gas_used)
}

// ============================================================
// Mathematical Functions
// ============================================================

/// Factorial: n! = n * (n-1) * ... * 1
/// C: int factorial(int n) { int r=1; while(n>0) { r*=n; n--; } return r; }
fn factorial(n: u32) -> (u32, u64) {
    let instructions = [
        addi(A0, ZERO, n as i32),   // a0 = n
        addi(T0, ZERO, 1),          // t0 = 1 (result)
        beq(A0, ZERO, 16),          // if n == 0, goto done
        mul(T0, T0, A0),            // t0 *= a0
        addi(A0, A0, -1),           // a0--
        jal(ZERO, -12),             // goto loop
        add(A0, T0, ZERO),          // a0 = t0
        ecall(),
    ];
    compile_and_run(&instructions)
}

/// Fibonacci: fib(n) = fib(n-1) + fib(n-2), fib(0)=0, fib(1)=1
/// C: int fib(int n) { int a=0,b=1,t; while(n>0) { t=a+b; a=b; b=t; n--; } return a; }
fn fibonacci(n: u32) -> (u32, u64) {
    // Layout: 0:addi 4:addi 8:addi 12:beq 16:add 20:add 24:add 28:addi 32:jal 36:add 40:ecall
    let instructions = [
        addi(A0, ZERO, n as i32),   // 0: a0 = n
        addi(T0, ZERO, 0),          // 4: t0 = a = 0
        addi(T1, ZERO, 1),          // 8: t1 = b = 1
        beq(A0, ZERO, 24),          // 12: if n == 0, goto done (offset 36)
        add(T2, T0, T1),            // 16: t2 = a + b
        add(T0, T1, ZERO),          // 20: a = b
        add(T1, T2, ZERO),          // 24: b = t2
        addi(A0, A0, -1),           // 28: n--
        jal(ZERO, -20),             // 32: goto beq (offset 12)
        add(A0, T0, ZERO),          // 36: a0 = a
        ecall(),                    // 40: return
    ];
    compile_and_run(&instructions)
}

/// GCD (Euclidean algorithm): gcd(a, b)
/// C: int gcd(int a, int b) { while(b!=0) { int t=b; b=a%b; a=t; } return a; }
fn gcd(a: u32, b: u32) -> (u32, u64) {
    let instructions = [
        addi(A0, ZERO, a as i32),   // a0 = a
        addi(A1, ZERO, b as i32),   // a1 = b
        beq(A1, ZERO, 20),          // if b == 0, goto done
        add(T0, A1, ZERO),          // t0 = b
        remu(A1, A0, A1),           // b = a % b
        add(A0, T0, ZERO),          // a = t0
        jal(ZERO, -16),             // goto loop
        ecall(),
    ];
    compile_and_run(&instructions)
}

/// Sum 1 to N: sum(n) = 1 + 2 + ... + n
/// C: int sum(int n) { int s=0; while(n>0) { s+=n; n--; } return s; }
fn sum_1_to_n(n: u32) -> (u32, u64) {
    let instructions = [
        addi(A0, ZERO, n as i32),   // a0 = n
        addi(T0, ZERO, 0),          // t0 = sum = 0
        beq(A0, ZERO, 16),          // if n == 0, goto done
        add(T0, T0, A0),            // sum += n
        addi(A0, A0, -1),           // n--
        jal(ZERO, -12),             // goto loop
        add(A0, T0, ZERO),          // a0 = sum
        ecall(),
    ];
    compile_and_run(&instructions)
}

/// Power: pow(base, exp) = base^exp
/// C: int pow(int b, int e) { int r=1; while(e>0) { r*=b; e--; } return r; }
fn power(base: u32, exp: u32) -> (u32, u64) {
    let instructions = [
        addi(A0, ZERO, base as i32), // a0 = base
        addi(A1, ZERO, exp as i32),  // a1 = exp
        addi(T0, ZERO, 1),           // t0 = result = 1
        beq(A1, ZERO, 16),           // if exp == 0, goto done
        mul(T0, T0, A0),             // result *= base
        addi(A1, A1, -1),            // exp--
        jal(ZERO, -12),              // goto loop
        add(A0, T0, ZERO),           // a0 = result
        ecall(),
    ];
    compile_and_run(&instructions)
}

/// Is Prime check (trial division)
/// C: int is_prime(int n) { if(n<2) return 0; for(int i=2; i*i<=n; i++) if(n%i==0) return 0; return 1; }
fn is_prime(n: u32) -> (u32, u64) {
    // Layout: 0:addi 4:addi 8:bge 12:addi 16:ecall 20:mul 24:bltu 28:remu 32:beq 36:addi 40:jal 44:addi 48:ecall 52:addi 56:ecall
    let instructions = [
        addi(A0, ZERO, n as i32),   // 0: a0 = n
        addi(T0, ZERO, 2),          // 4: t0 = i = 2
        // Check n < 2
        bge(A0, T0, 12),            // 8: if n >= 2, goto loop_check at 20
        addi(A0, ZERO, 0),          // 12: return 0 (n < 2)
        ecall(),                    // 16
        // Loop check: while i*i <= n
        mul(T1, T0, T0),            // 20: t1 = i * i
        bltu(A0, T1, 28),           // 24: if n < i*i, goto is_prime at 52
        remu(T2, A0, T0),           // 28: t2 = n % i
        beq(T2, ZERO, 12),          // 32: if n%i == 0, goto not_prime at 44
        addi(T0, T0, 1),            // 36: i++
        jal(ZERO, -20),             // 40: goto loop_check at 20
        addi(A0, ZERO, 0),          // 44: not_prime: return 0
        ecall(),                    // 48
        addi(A0, ZERO, 1),          // 52: is_prime: return 1
        ecall(),                    // 56
    ];
    compile_and_run(&instructions)
}

fn get_base_gas() -> u64 {
    // Minimal program: just return 0
    let instructions = [
        addi(A0, ZERO, 0),
        ecall(),
    ];
    compile_and_run(&instructions).1
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        Mathematical Functions Benchmark: C → rv32im → EVM                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let base_gas = get_base_gas();
    println!("Base gas (minimal program): {}\n", base_gas);

    // Factorial
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ FACTORIAL: n! = n × (n-1) × ... × 1                                         │");
    println!("├────────────────────┬────────────────┬────────────────┬──────────────────────┤");
    println!("│ Input              │ Result         │ Total Gas      │ Exec Gas (- base)    │");
    println!("├────────────────────┼────────────────┼────────────────┼──────────────────────┤");
    for n in [3, 5, 7, 10] {
        let (result, gas) = factorial(n);
        let exec_gas = gas.saturating_sub(base_gas);
        println!("│ factorial({:2})      │ {:>14} │ {:>14} │ {:>20} │", n, result, gas, exec_gas);
    }
    println!("└────────────────────┴────────────────┴────────────────┴──────────────────────┘\n");

    // Fibonacci
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ FIBONACCI: fib(n) = fib(n-1) + fib(n-2)                                     │");
    println!("├────────────────────┬────────────────┬────────────────┬──────────────────────┤");
    println!("│ Input              │ Result         │ Total Gas      │ Exec Gas (- base)    │");
    println!("├────────────────────┼────────────────┼────────────────┼──────────────────────┤");
    for n in [5, 10, 15, 20] {
        let (result, gas) = fibonacci(n);
        let exec_gas = gas.saturating_sub(base_gas);
        println!("│ fibonacci({:2})      │ {:>14} │ {:>14} │ {:>20} │", n, result, gas, exec_gas);
    }
    println!("└────────────────────┴────────────────┴────────────────┴──────────────────────┘\n");

    // GCD
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ GCD: Greatest Common Divisor (Euclidean algorithm)                          │");
    println!("├────────────────────┬────────────────┬────────────────┬──────────────────────┤");
    println!("│ Input              │ Result         │ Total Gas      │ Exec Gas (- base)    │");
    println!("├────────────────────┼────────────────┼────────────────┼──────────────────────┤");
    for (a, b) in [(48, 18), (100, 35), (1071, 462), (10000, 7777)] {
        let (result, gas) = gcd(a, b);
        let exec_gas = gas.saturating_sub(base_gas);
        println!("│ gcd({:5}, {:5})   │ {:>14} │ {:>14} │ {:>20} │", a, b, result, gas, exec_gas);
    }
    println!("└────────────────────┴────────────────┴────────────────┴──────────────────────┘\n");

    // Sum 1 to N
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SUM: 1 + 2 + ... + n                                                        │");
    println!("├────────────────────┬────────────────┬────────────────┬──────────────────────┤");
    println!("│ Input              │ Result         │ Total Gas      │ Exec Gas (- base)    │");
    println!("├────────────────────┼────────────────┼────────────────┼──────────────────────┤");
    for n in [10, 50, 100, 1000] {
        let (result, gas) = sum_1_to_n(n);
        let exec_gas = gas.saturating_sub(base_gas);
        println!("│ sum(1..{:4})        │ {:>14} │ {:>14} │ {:>20} │", n, result, gas, exec_gas);
    }
    println!("└────────────────────┴────────────────┴────────────────┴──────────────────────┘\n");

    // Power
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ POWER: base^exp                                                             │");
    println!("├────────────────────┬────────────────┬────────────────┬──────────────────────┤");
    println!("│ Input              │ Result         │ Total Gas      │ Exec Gas (- base)    │");
    println!("├────────────────────┼────────────────┼────────────────┼──────────────────────┤");
    for (base, exp) in [(2, 10), (3, 7), (5, 5), (7, 4)] {
        let (result, gas) = power(base, exp);
        let exec_gas = gas.saturating_sub(base_gas);
        println!("│ pow({}, {:2})          │ {:>14} │ {:>14} │ {:>20} │", base, exp, result, gas, exec_gas);
    }
    println!("└────────────────────┴────────────────┴────────────────┴──────────────────────┘\n");

    // Is Prime
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ IS_PRIME: Trial division primality test                                     │");
    println!("├────────────────────┬────────────────┬────────────────┬──────────────────────┤");
    println!("│ Input              │ Result         │ Total Gas      │ Exec Gas (- base)    │");
    println!("├────────────────────┼────────────────┼────────────────┼──────────────────────┤");
    for n in [7, 17, 97, 100, 997] {
        let (result, gas) = is_prime(n);
        let exec_gas = gas.saturating_sub(base_gas);
        let prime_str = if result == 1 { "prime" } else { "composite" };
        println!("│ is_prime({:4})      │ {:>14} │ {:>14} │ {:>20} │", n, prime_str, gas, exec_gas);
    }
    println!("└────────────────────┴────────────────┴────────────────┴──────────────────────┘\n");

    // Summary
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SUMMARY: Gas cost per iteration (approximate)                               │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");

    let (_, gas1) = factorial(1);
    let (_, gas10) = factorial(10);
    let factorial_per_iter = (gas10 - gas1) / 9;
    println!("│ Factorial loop iteration:    ~{:4} gas                                      │", factorial_per_iter);

    let (_, gas1) = fibonacci(1);
    let (_, gas10) = fibonacci(10);
    let fib_per_iter = (gas10 - gas1) / 9;
    println!("│ Fibonacci loop iteration:    ~{:4} gas                                      │", fib_per_iter);

    let (_, gas1) = sum_1_to_n(1);
    let (_, gas100) = sum_1_to_n(100);
    let sum_per_iter = (gas100 - gas1) / 99;
    println!("│ Sum loop iteration:          ~{:4} gas                                      │", sum_per_iter);

    println!("└──────────────────────────────────────────────────────────────────────────────┘");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial_3() {
        let (result, _) = factorial(3);
        assert_eq!(result, 6);
    }

    #[test]
    fn test_factorial_5() {
        let (result, _) = factorial(5);
        assert_eq!(result, 120);
    }

    #[test]
    fn test_factorial_7() {
        let (result, _) = factorial(7);
        assert_eq!(result, 5040);
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0).0, 0);
        assert_eq!(fibonacci(1).0, 1);
        assert_eq!(fibonacci(10).0, 55);
        assert_eq!(fibonacci(20).0, 6765);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18).0, 6);
        assert_eq!(gcd(100, 35).0, 5);
        assert_eq!(gcd(1071, 462).0, 21);
    }

    #[test]
    fn test_sum() {
        assert_eq!(sum_1_to_n(10).0, 55);
        assert_eq!(sum_1_to_n(100).0, 5050);
    }

    #[test]
    fn test_power() {
        assert_eq!(power(2, 10).0, 1024);
        assert_eq!(power(3, 7).0, 2187);
    }

    #[test]
    fn test_is_prime() {
        assert_eq!(is_prime(2).0, 1);
        assert_eq!(is_prime(7).0, 1);
        assert_eq!(is_prime(17).0, 1);
        assert_eq!(is_prime(97).0, 1);
        assert_eq!(is_prime(4).0, 0);
        assert_eq!(is_prime(100).0, 0);
    }
}
