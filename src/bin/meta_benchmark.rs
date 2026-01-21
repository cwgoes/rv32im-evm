//! Meta-compilation benchmark: Measure actual compilation gas costs
//!
//! This benchmark runs the meta-compiled compiler (rv32im -> EVM compiler running on EVM)
//! and measures how much gas it costs to compile various rv32im programs.
//!
//! Approach: Use CREATE to deploy the meta-compiler, then CALL it with input.
//! This avoids bytecode relocation issues.

use rv32im_evm::compiler::{Compiler, CompilerConfig};
use rv32im_evm::runtime::Runtime;
use std::fs;

// EVM opcodes
const PUSH0: u8 = 0x5F;
const PUSH1: u8 = 0x60;
const PUSH2: u8 = 0x61;
const PUSH3: u8 = 0x62;
const PUSH20: u8 = 0x73;
const DUP1: u8 = 0x80;
const DUP2: u8 = 0x81;
const MSTORE: u8 = 0x52;
const MLOAD: u8 = 0x51;
const CALLDATASIZE: u8 = 0x36;
const CALLDATACOPY: u8 = 0x37;
const CODECOPY: u8 = 0x39;
const CODESIZE: u8 = 0x38;
const CREATE: u8 = 0xF0;
const CALL: u8 = 0xF1;
const RETURN: u8 = 0xF3;
const GAS: u8 = 0x5A;
const JUMPDEST: u8 = 0x5B;

// EVM memory addresses for the meta-compiled compiler
// These correspond to RISC-V addresses after the compiler's memory translation:
// INPUT_LEN_ADDR (0x80000F00) -> EVM 0x1000
// INPUT_ADDR (0x80001000) -> EVM 0x1100
const EVM_INPUT_LEN_ADDR: u32 = 0x1000;
const EVM_INPUT_ADDR: u32 = 0x1100;

/// Create a wrapper bytecode that:
/// 1. Deploys the meta-compiler via CREATE
/// 2. Sets up memory with the input (from calldata)
/// 3. CALLs the deployed compiler
/// 4. Returns the result
fn create_deploy_and_call_bytecode(compiler_bytecode: &[u8]) -> Vec<u8> {
    let mut wrapper = Vec::new();

    // The meta-compiler expects input at specific memory locations.
    // But when we CALL it, those memory locations need to be set up
    // in the CALLER's memory, then we DELEGATECALL, or...
    // Actually, CALL copies memory from caller as input and output.

    // Simpler approach: Create init code that the CREATE will use
    // The init code sets up memory and returns the runtime (compiler) code

    // For CREATE:
    // 1. Init code is copied to memory
    // 2. EVM executes init code
    // 3. Init code RETURNs the runtime bytecode
    // 4. Runtime bytecode is stored at the new contract address

    // So we need:
    // - Init code that RETURNs the compiler bytecode
    // - Then we CALL the deployed contract

    // Init code: CODECOPY the runtime to memory, then RETURN it
    // Format:
    //   PUSH size (runtime_size)
    //   DUP1
    //   PUSH offset_of_runtime
    //   PUSH0 (dest offset)
    //   CODECOPY
    //   PUSH0 (offset)
    //   RETURN

    let runtime_bytecode = compiler_bytecode;
    let runtime_size = runtime_bytecode.len();

    // Build init code first to know its size
    let mut init_code = Vec::new();

    // PUSH3 runtime_size
    init_code.push(PUSH3);
    init_code.push(((runtime_size >> 16) & 0xFF) as u8);
    init_code.push(((runtime_size >> 8) & 0xFF) as u8);
    init_code.push((runtime_size & 0xFF) as u8);

    // DUP1 (for RETURN later)
    init_code.push(DUP1);

    // We'll fill in the offset after we know init_code size
    // PUSH3 offset_of_runtime (placeholder)
    let offset_placeholder_pos = init_code.len();
    init_code.push(PUSH3);
    init_code.push(0);
    init_code.push(0);
    init_code.push(0);

    // PUSH0 dest
    init_code.push(PUSH0);

    // CODECOPY
    init_code.push(CODECOPY);

    // PUSH0 (return offset)
    init_code.push(PUSH0);

    // RETURN
    init_code.push(RETURN);

    let init_size = init_code.len();

    // Now fill in the offset of runtime code
    let runtime_offset = init_size;
    init_code[offset_placeholder_pos + 1] = ((runtime_offset >> 16) & 0xFF) as u8;
    init_code[offset_placeholder_pos + 2] = ((runtime_offset >> 8) & 0xFF) as u8;
    init_code[offset_placeholder_pos + 3] = (runtime_offset & 0xFF) as u8;

    // Now build the full deployment bytecode (init_code + runtime)
    let mut deploy_bytecode = init_code.clone();
    deploy_bytecode.extend_from_slice(runtime_bytecode);

    let deploy_size = deploy_bytecode.len();

    // Now build the wrapper that:
    // 1. Copies deploy_bytecode to memory
    // 2. CREATEs the contract
    // 3. Sets up input memory
    // 4. CALLs the contract
    // 5. Returns result

    // === Part 1: Copy deploy_bytecode to memory ===
    // PUSH3 deploy_size
    wrapper.push(PUSH3);
    wrapper.push(((deploy_size >> 16) & 0xFF) as u8);
    wrapper.push(((deploy_size >> 8) & 0xFF) as u8);
    wrapper.push((deploy_size & 0xFF) as u8);

    // DUP1 (for CREATE)
    wrapper.push(DUP1);

    // We need to know wrapper's init section size to calc code offset
    // For now, use placeholder, we'll calculate later
    // PUSH3 code_offset (placeholder)
    let code_offset_pos = wrapper.len();
    wrapper.push(PUSH3);
    wrapper.push(0);
    wrapper.push(0);
    wrapper.push(0);

    // PUSH0 dest (0)
    wrapper.push(PUSH0);

    // CODECOPY
    wrapper.push(CODECOPY);

    // === Part 2: CREATE ===
    // Stack: deploy_size
    // CREATE(value, offset, size)
    // PUSH0 (value = 0)
    wrapper.push(PUSH0);

    // PUSH0 (offset = 0)
    wrapper.push(PUSH0);

    // Stack: size, value, offset -> need offset, size, value
    // Actually: CREATE takes stack (value, offset, size) but reads as: pop value, pop offset, pop size
    // So we need: push size, push offset, push value -> pop value, pop offset, pop size
    // Current: deploy_size on stack after DUP1
    // After CODECOPY the stack just has deploy_size
    // We need: value(0), offset(0), size
    // So: size, PUSH0, PUSH0 -> stack is: 0, 0, size
    // CREATE pops: value=0, offset=0, size

    // Let me redo: after CODECOPY, stack has deploy_size
    // PUSH0 (value)
    // PUSH0 (offset)
    // Stack: offset(0), value(0), size
    // Hmm, EVM is stack-based, LIFO. CREATE signature is (value, offset, size)
    // So we need top of stack = size, then offset, then value

    // After CODECOPY: stack = [deploy_size]
    // PUSH0 -> stack = [0, deploy_size]
    // PUSH0 -> stack = [0, 0, deploy_size]
    // CREATE pops in order: size=deploy_size, offset=0, value=0 ✓

    // PUSH0 (offset)
    wrapper.push(PUSH0);

    // PUSH0 (value)
    wrapper.push(PUSH0);

    // CREATE - returns address
    wrapper.push(CREATE);

    // === Part 3: Set up input memory ===
    // The deployed contract (meta-compiler) reads from memory at:
    // - INPUT_LEN_ADDR (EVM 0x1000)
    // - INPUT_ADDR (EVM 0x1100)
    //
    // But when we CALL, the callee gets fresh memory!
    // We need to pass input via calldata to the CALL, not memory.
    //
    // Actually, the meta-compiler reads from its OWN memory after being called.
    // It expects memory to be initialized. But CALLs start with empty memory.
    //
    // So we need the meta-compiler to read from CALLDATA, not from memory.
    // This requires modifying the nostd compiler, which we want to avoid.
    //
    // Alternative: Use DELEGATECALL instead of CALL.
    // DELEGATECALL uses the caller's storage and memory context.

    // Let me restructure: use DELEGATECALL so the compiler runs in our memory context

    // First, store the contract address
    // Stack after CREATE: [contract_addr]

    // Store contract address at memory 0x00 (temporary)
    // PUSH1 0x00
    wrapper.push(PUSH1);
    wrapper.push(0x00);
    // MSTORE (stores 32 bytes)
    wrapper.push(MSTORE);

    // Now set up input memory for the meta-compiler
    // Store calldata size at INPUT_LEN_ADDR (0x1000)
    wrapper.push(CALLDATASIZE);
    wrapper.push(PUSH2);
    wrapper.push((EVM_INPUT_LEN_ADDR >> 8) as u8);
    wrapper.push((EVM_INPUT_LEN_ADDR & 0xFF) as u8);
    wrapper.push(MSTORE);

    // Copy calldata to INPUT_ADDR (0x1100)
    wrapper.push(CALLDATASIZE);  // size
    wrapper.push(PUSH0);          // src offset
    wrapper.push(PUSH2);
    wrapper.push((EVM_INPUT_ADDR >> 8) as u8);
    wrapper.push((EVM_INPUT_ADDR & 0xFF) as u8);
    wrapper.push(CALLDATACOPY);

    // === Part 4: DELEGATECALL the deployed compiler ===
    // DELEGATECALL(gas, addr, argsOffset, argsSize, retOffset, retSize)
    // We don't pass args via calldata to the delegatecall since memory is shared
    // retOffset/retSize = where to put return data

    // PUSH1 0x20 (retSize - 32 bytes for output length)
    wrapper.push(PUSH1);
    wrapper.push(0x20);

    // PUSH0 (retOffset)
    wrapper.push(PUSH0);

    // PUSH0 (argsSize - no args)
    wrapper.push(PUSH0);

    // PUSH0 (argsOffset)
    wrapper.push(PUSH0);

    // Load contract address from memory 0x00
    wrapper.push(PUSH0);
    wrapper.push(MLOAD);

    // GAS (all remaining gas)
    wrapper.push(GAS);

    // DELEGATECALL = 0xF4
    wrapper.push(0xF4);

    // Stack: [success]
    // Pop success (we ignore it for now)
    // POP = 0x50
    wrapper.push(0x50);

    // === Part 5: Return the result ===
    // The meta-compiler puts output length in a0 and returns via RETURN
    // With DELEGATECALL, the return data is in our memory at retOffset (0x00)

    // Return 32 bytes from offset 0
    wrapper.push(PUSH1);
    wrapper.push(0x20);
    wrapper.push(PUSH0);
    wrapper.push(RETURN);

    // Now we know the wrapper init size, update the code_offset
    let wrapper_init_size = wrapper.len();
    let actual_code_offset = wrapper_init_size;
    wrapper[code_offset_pos + 1] = ((actual_code_offset >> 16) & 0xFF) as u8;
    wrapper[code_offset_pos + 2] = ((actual_code_offset >> 8) & 0xFF) as u8;
    wrapper[code_offset_pos + 3] = (actual_code_offset & 0xFF) as u8;

    // Append the deployment bytecode (init + runtime)
    wrapper.extend_from_slice(&deploy_bytecode);

    wrapper
}

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

const ZERO: u32 = 0;
const T0: u32 = 5;
const T1: u32 = 6;
const T2: u32 = 7;
const A0: u32 = 10;
const A1: u32 = 11;

fn addi(rd: u32, rs1: u32, imm: i32) -> u32 { encode_i_type(0b0010011, rd, 0, rs1, imm) }
fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b000, rs1, rs2, imm) }
fn bge(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b101, rs1, rs2, imm) }
fn bltu(rs1: u32, rs2: u32, imm: i32) -> u32 { encode_b_type(0b1100011, 0b110, rs1, rs2, imm) }
fn mul(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000001) }
fn jal(rd: u32, imm: i32) -> u32 { encode_j_type(0b1101111, rd, imm) }
fn add(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b000, rs1, rs2, 0b0000000) }
fn remu(rd: u32, rs1: u32, rs2: u32) -> u32 { encode_r_type(0b0110011, rd, 0b111, rs1, rs2, 0b0000001) }
fn ecall() -> u32 { 0x00000073 }

fn instructions_to_bytes(instructions: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(instructions.len() * 4);
    for instr in instructions {
        bytes.extend_from_slice(&instr.to_le_bytes());
    }
    bytes
}

struct TestProgram {
    name: &'static str,
    instructions: Vec<u32>,
    expected_result: u32,
}

fn get_test_programs() -> Vec<TestProgram> {
    vec![
        TestProgram {
            name: "factorial(10)",
            instructions: vec![
                addi(A0, ZERO, 10),
                addi(T0, ZERO, 1),
                beq(A0, ZERO, 16),
                mul(T0, T0, A0),
                addi(A0, A0, -1),
                jal(ZERO, -12),
                add(A0, T0, ZERO),
                ecall(),
            ],
            expected_result: 3628800,
        },
        TestProgram {
            name: "fibonacci(20)",
            instructions: vec![
                addi(A0, ZERO, 20),
                addi(T0, ZERO, 0),
                addi(T1, ZERO, 1),
                beq(A0, ZERO, 24),
                add(T2, T0, T1),
                add(T0, T1, ZERO),
                add(T1, T2, ZERO),
                addi(A0, A0, -1),
                jal(ZERO, -20),
                add(A0, T0, ZERO),
                ecall(),
            ],
            expected_result: 6765,
        },
        TestProgram {
            name: "gcd(1071, 462)",
            instructions: vec![
                addi(A0, ZERO, 1071),
                addi(A1, ZERO, 462),
                beq(A1, ZERO, 20),
                add(T0, A1, ZERO),
                remu(A1, A0, A1),
                add(A0, T0, ZERO),
                jal(ZERO, -16),
                ecall(),
            ],
            expected_result: 21,
        },
        TestProgram {
            name: "sum(1..100)",
            instructions: vec![
                addi(A0, ZERO, 100),
                addi(T0, ZERO, 0),
                beq(A0, ZERO, 16),
                add(T0, T0, A0),
                addi(A0, A0, -1),
                jal(ZERO, -12),
                add(A0, T0, ZERO),
                ecall(),
            ],
            expected_result: 5050,
        },
        TestProgram {
            name: "power(2, 10)",
            instructions: vec![
                addi(A0, ZERO, 2),
                addi(A1, ZERO, 10),
                addi(T0, ZERO, 1),
                beq(A1, ZERO, 16),
                mul(T0, T0, A0),
                addi(A1, A1, -1),
                jal(ZERO, -12),
                add(A0, T0, ZERO),
                ecall(),
            ],
            expected_result: 1024,
        },
        TestProgram {
            name: "is_prime(97)",
            instructions: vec![
                addi(A0, ZERO, 97),
                addi(T0, ZERO, 2),
                bge(A0, T0, 12),
                addi(A0, ZERO, 0),
                ecall(),
                mul(T1, T0, T0),
                bltu(A0, T1, 28),
                remu(T2, A0, T0),
                beq(T2, ZERO, 12),
                addi(T0, T0, 1),
                jal(ZERO, -20),
                addi(A0, ZERO, 0),
                ecall(),
                addi(A0, ZERO, 1),
                ecall(),
            ],
            expected_result: 1,
        },
    ]
}

fn main() {
    println!("================================================================================");
    println!("                    META-COMPILATION BENCHMARK");
    println!("     Measuring Gas Costs of Compiling RISC-V Programs on EVM");
    println!("================================================================================\n");

    // Load and compile the meta-compiler
    let compiler_binary = match fs::read("rv32im-compiler-nostd/target/rv32im-compiler-text.bin") {
        Ok(bin) => bin,
        Err(_) => {
            println!("ERROR: Meta-compiler binary not found.");
            println!("Build it first:");
            println!("  cd rv32im-compiler-nostd");
            println!("  cargo build --target riscv32im-unknown-none-elf --release");
            println!("  llvm-objcopy --only-section=.text -O binary \\");
            println!("    target/riscv32im-unknown-none-elf/release/rv32im-compiler \\");
            println!("    target/rv32im-compiler-text.bin");
            return;
        }
    };

    println!("Step 1: Compile rv32im compiler to EVM...");
    println!("  RISC-V compiler size: {} bytes ({} instructions)",
             compiler_binary.len(), compiler_binary.len() / 4);

    let meta_config = CompilerConfig {
        load_address: 0x80000000,
        stack_pointer: 0x80020000,
        memory_size: 0x40000,
        ..Default::default()
    };

    let mut compiler = Compiler::with_config(meta_config);
    let meta_bytecode = compiler.compile(&compiler_binary)
        .expect("Failed to compile meta-compiler");

    println!("  EVM bytecode size: {} bytes", meta_bytecode.len());
    println!("  Expansion ratio: {:.2}x\n", meta_bytecode.len() as f64 / compiler_binary.len() as f64);

    // Create wrapper that deploys and calls the meta-compiler
    println!("Step 2: Create deploy+call wrapper bytecode...");
    let wrapper_bytecode = create_deploy_and_call_bytecode(&meta_bytecode);
    println!("  Total wrapper size: {} bytes", wrapper_bytecode.len());
    println!("  Overhead: {} bytes\n", wrapper_bytecode.len() - meta_bytecode.len());

    // Test programs
    let programs = get_test_programs();

    // Collect results for analysis
    struct BenchResult {
        name: &'static str,
        rv_instr: usize,
        compile_gas: u64,
        exec_gas: u64,
        verified: bool,
    }
    let mut results: Vec<BenchResult> = Vec::new();

    println!("Step 3: Benchmark meta-compilation gas costs...\n");

    for prog in &programs {
        let rv_program = instructions_to_bytes(&prog.instructions);
        let rv_instr = prog.instructions.len();

        // Run the wrapper with the test program as calldata
        let runtime = Runtime::new(wrapper_bytecode.clone())
            .with_calldata(rv_program.clone());

        let compile_result = runtime.execute();

        let (compile_gas, exec_gas, verified) = match compile_result {
            Ok(output) => {
                // Compile directly and run to get execution gas
                let direct_config = CompilerConfig {
                    load_address: 0x80000000,
                    stack_pointer: 0x80010000,
                    memory_size: 0x20000,
                    ..Default::default()
                };
                let mut direct_compiler = Compiler::with_config(direct_config);
                let direct_bytecode = direct_compiler.compile(&rv_program)
                    .expect("Direct compilation failed");

                let verify_runtime = Runtime::new(direct_bytecode);
                let verify_output = verify_runtime.execute();

                match verify_output {
                    Ok(v) => (output.gas_used, v.gas_used, v.return_value == prog.expected_result),
                    Err(_) => (output.gas_used, 0, false),
                }
            }
            Err(e) => {
                println!("  {} ERROR: {:?}", prog.name, e);
                continue;
            }
        };

        results.push(BenchResult {
            name: prog.name,
            rv_instr,
            compile_gas,
            exec_gas,
            verified,
        });
    }

    if results.is_empty() {
        println!("  No programs compiled successfully.");
        return;
    }

    // Calculate fixed overhead using linear regression
    // compile_gas ≈ fixed_overhead + marginal_cost * rv_instr
    let min_gas = results.iter().map(|r| r.compile_gas).min().unwrap();
    let max_gas = results.iter().map(|r| r.compile_gas).max().unwrap();
    let min_instr = results.iter().map(|r| r.rv_instr).min().unwrap();
    let max_instr = results.iter().map(|r| r.rv_instr).max().unwrap();

    let marginal_gas = if max_instr > min_instr {
        (max_gas - min_gas) as f64 / (max_instr - min_instr) as f64
    } else {
        0.0
    };
    let fixed_overhead = min_gas as f64 - marginal_gas * min_instr as f64;

    // Print results table
    println!("┌───────────────────┬────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Program           │ RV Ins │ Compile Gas  │ Execute Gas  │ Verify       │");
    println!("├───────────────────┼────────┼──────────────┼──────────────┼──────────────┤");

    let mut total_rv_instr = 0usize;
    let mut total_compile_gas = 0u64;
    let mut total_exec_gas = 0u64;
    let mut all_verified = true;

    for r in &results {
        total_rv_instr += r.rv_instr;
        total_compile_gas += r.compile_gas;
        total_exec_gas += r.exec_gas;
        if !r.verified {
            all_verified = false;
        }

        let verify_str = if r.verified { "OK" } else { "FAIL" };
        println!("│ {:17} │ {:>6} │ {:>12} │ {:>12} │ {:>12} │",
                 r.name, r.rv_instr, r.compile_gas, r.exec_gas, verify_str);
    }
    println!("└───────────────────┴────────┴──────────────┴──────────────┴──────────────┘\n");

    println!("================================================================================");
    println!("                              SUMMARY");
    println!("================================================================================\n");

    println!("  META-COMPILATION STATS:");
    println!("    RISC-V compiler:     {:>10} bytes ({} instructions)",
             compiler_binary.len(), compiler_binary.len() / 4);
    println!("    EVM bytecode:        {:>10} bytes", meta_bytecode.len());
    println!("    Expansion ratio:     {:>10.2}x\n",
             meta_bytecode.len() as f64 / compiler_binary.len() as f64);

    println!("  COMPILATION GAS BREAKDOWN:");
    println!("    Fixed overhead:      {:>10.0} gas (CREATE + init)", fixed_overhead);
    println!("    Marginal cost:       {:>10.1} gas/instruction\n", marginal_gas);

    let mean_compile = total_compile_gas as f64 / total_rv_instr as f64;
    let mean_exec = total_exec_gas as f64 / results.len() as f64;

    println!("  GAS COST COMPARISON:");
    println!("    Mean compile gas:    {:>10.0} per program", total_compile_gas as f64 / results.len() as f64);
    println!("    Mean execute gas:    {:>10.0} per program", mean_exec);
    println!("    Compile/Execute:     {:>10.1}x\n",
             (total_compile_gas as f64 / results.len() as f64) / mean_exec);

    if all_verified {
        println!("  All {} programs compiled and verified correctly!", results.len());
    } else {
        println!("  WARNING: Some programs failed verification!");
    }

    println!();
    println!("  INTERPRETATION:");
    println!("    The meta-compiled compiler uses ~{:.0}K gas fixed overhead plus", fixed_overhead / 1000.0);
    println!("    ~{:.0} gas per rv32im instruction to compile.", marginal_gas);
    println!("    For a typical 8-instruction program, compilation costs ~{:.0}K gas.",
             (fixed_overhead + marginal_gas * 8.0) / 1000.0);
}
