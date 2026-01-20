/*
 * Mini rv32im-to-EVM compiler
 *
 * A minimal compiler that can compile simple rv32im programs to EVM bytecode.
 * Designed to be compiled to rv32im itself for meta-compilation experiments.
 *
 * Supports: ADDI, ADD, MUL, BEQ, BNE, JAL, ECALL
 *
 * Memory Layout (when running on EVM):
 * - Registers: 0x0000 - 0x03FF (32 regs * 32 bytes)
 * - PC: 0x0400
 * - Return value: 0x0420
 * - RISC-V memory: 0x0500+
 *
 * Input/Output convention:
 * - Input program at address INPUT_ADDR
 * - Input length (in bytes) at INPUT_LEN_ADDR
 * - Output bytecode written to OUTPUT_ADDR
 * - Output length stored at OUTPUT_LEN_ADDR
 */

#include <stdint.h>

/* Memory addresses for input/output (in RISC-V address space) */
/* Keep addresses close to code to minimize EVM memory expansion */
#define LOAD_ADDRESS    0x80000000
#define INPUT_LEN_ADDR  0x80000C00   /* Input program length (at 3KB offset) */
#define INPUT_ADDR      0x80000C10   /* Input program starts here */
#define OUTPUT_LEN_ADDR 0x80000D00   /* Output bytecode length (at 3.25KB) */
#define OUTPUT_ADDR     0x80000D10   /* Output bytecode starts here */

/* EVM Opcodes */
#define OP_STOP       0x00
#define OP_ADD        0x01
#define OP_MUL        0x02
#define OP_SUB        0x03
#define OP_DIV        0x04
#define OP_MOD        0x06
#define OP_LT         0x10
#define OP_GT         0x11
#define OP_SLT        0x12
#define OP_EQ         0x14
#define OP_ISZERO     0x15
#define OP_AND        0x16
#define OP_OR         0x17
#define OP_XOR        0x18
#define OP_NOT        0x19
#define OP_SHL        0x1b
#define OP_SHR        0x1c
#define OP_SAR        0x1d
#define OP_POP        0x50
#define OP_MLOAD      0x51
#define OP_MSTORE     0x52
#define OP_JUMP       0x56
#define OP_JUMPI      0x57
#define OP_JUMPDEST   0x5b
#define OP_PUSH0      0x5f
#define OP_PUSH1      0x60
#define OP_PUSH2      0x61
#define OP_PUSH4      0x63
#define OP_DUP1       0x80
#define OP_SWAP1      0x90
#define OP_RETURN     0xf3

/* Register and memory constants */
#define REG_BASE      0x0000
#define REG_SIZE      32
#define PC_ADDR       0x0400
#define RETVAL_ADDR   0x0420
#define MEM_BASE      0x0500

/* Compiler state */
static uint8_t *output;
static uint32_t out_pos;
static uint32_t pc_to_evm[64];  /* Map RISC-V PC offset to EVM position */
static uint32_t pending_jumps[32][2];  /* [jump_pos, target_pc] */
static uint32_t num_pending;

/* Emit single byte */
static void emit(uint8_t b) {
    output[out_pos++] = b;
}

/* Emit PUSH1 value */
static void push1(uint8_t val) {
    emit(OP_PUSH1);
    emit(val);
}

/* Emit PUSH2 value */
static void push2(uint16_t val) {
    emit(OP_PUSH2);
    emit((val >> 8) & 0xFF);
    emit(val & 0xFF);
}

/* Emit PUSH4 value */
static void push4(uint32_t val) {
    emit(OP_PUSH4);
    emit((val >> 24) & 0xFF);
    emit((val >> 16) & 0xFF);
    emit((val >> 8) & 0xFF);
    emit(val & 0xFF);
}

/* Push optimal size for u32 value */
static void push_u32(uint32_t val) {
    if (val == 0) {
        emit(OP_PUSH0);
    } else if (val <= 0xFF) {
        push1((uint8_t)val);
    } else if (val <= 0xFFFF) {
        push2((uint16_t)val);
    } else {
        push4(val);
    }
}

/* Load register onto stack */
static void load_reg(uint8_t reg) {
    if (reg == 0) {
        emit(OP_PUSH0);
    } else {
        uint32_t addr = REG_BASE + reg * REG_SIZE;
        push_u32(addr);
        emit(OP_MLOAD);
    }
}

/* Store top of stack to register */
static void store_reg(uint8_t reg) {
    if (reg == 0) {
        emit(OP_POP);
    } else {
        uint32_t addr = REG_BASE + reg * REG_SIZE;
        push_u32(addr);
        emit(OP_MSTORE);
    }
}

/* Mask to 32 bits */
static void mask32(void) {
    push4(0xFFFFFFFF);
    emit(OP_AND);
}

/* Extract instruction fields */
static uint8_t get_rd(uint32_t instr) { return (instr >> 7) & 0x1F; }
static uint8_t get_rs1(uint32_t instr) { return (instr >> 15) & 0x1F; }
static uint8_t get_rs2(uint32_t instr) { return (instr >> 20) & 0x1F; }
static uint8_t get_funct3(uint32_t instr) { return (instr >> 12) & 0x7; }
static uint8_t get_funct7(uint32_t instr) { return (instr >> 25) & 0x7F; }

static int32_t get_imm_i(uint32_t instr) {
    int32_t imm = (int32_t)instr >> 20;
    return imm;
}

static int32_t get_imm_b(uint32_t instr) {
    uint32_t imm11 = (instr >> 7) & 1;
    uint32_t imm4_1 = (instr >> 8) & 0xF;
    uint32_t imm10_5 = (instr >> 25) & 0x3F;
    uint32_t imm12 = (instr >> 31) & 1;
    int32_t imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
    if (imm12) imm |= 0xFFFFE000;  /* Sign extend */
    return imm;
}

static int32_t get_imm_j(uint32_t instr) {
    uint32_t imm20 = (instr >> 31) & 1;
    uint32_t imm10_1 = (instr >> 21) & 0x3FF;
    uint32_t imm11 = (instr >> 20) & 1;
    uint32_t imm19_12 = (instr >> 12) & 0xFF;
    int32_t imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
    if (imm20) imm |= 0xFFE00000;  /* Sign extend */
    return imm;
}

/* Add pending jump to be resolved later */
static void add_pending_jump(uint32_t jump_pos, uint32_t target_pc) {
    pending_jumps[num_pending][0] = jump_pos;
    pending_jumps[num_pending][1] = target_pc;
    num_pending++;
}

/* Compile a single instruction */
static void compile_instruction(uint32_t pc, uint32_t instr) {
    uint8_t opcode = instr & 0x7F;
    uint8_t rd = get_rd(instr);
    uint8_t rs1 = get_rs1(instr);
    uint8_t rs2 = get_rs2(instr);
    uint8_t funct3 = get_funct3(instr);
    uint8_t funct7 = get_funct7(instr);

    switch (opcode) {
    case 0x13:  /* I-type arithmetic */
        if (funct3 == 0) {  /* ADDI */
            if (rd != 0) {
                int32_t imm = get_imm_i(instr);
                load_reg(rs1);
                if (imm != 0) {
                    push_u32((uint32_t)imm);
                    emit(OP_ADD);
                    mask32();
                }
                store_reg(rd);
            }
        }
        break;

    case 0x33:  /* R-type */
        if (rd != 0) {
            if (funct7 == 0x00 && funct3 == 0) {  /* ADD */
                load_reg(rs1);
                load_reg(rs2);
                emit(OP_ADD);
                mask32();
                store_reg(rd);
            } else if (funct7 == 0x01 && funct3 == 0) {  /* MUL */
                load_reg(rs1);
                load_reg(rs2);
                emit(OP_MUL);
                mask32();
                store_reg(rd);
            }
        }
        break;

    case 0x63:  /* B-type branches */
        {
            int32_t imm = get_imm_b(instr);
            uint32_t target_pc = pc + imm;

            if (funct3 == 0) {  /* BEQ */
                if (rs2 == 0) {
                    load_reg(rs1);
                    emit(OP_ISZERO);
                } else if (rs1 == 0) {
                    load_reg(rs2);
                    emit(OP_ISZERO);
                } else {
                    load_reg(rs1);
                    load_reg(rs2);
                    emit(OP_EQ);
                }
                /* JUMPI setup */
                emit(OP_PUSH2);
                add_pending_jump(out_pos, target_pc);
                emit(0); emit(0);  /* Placeholder */
                emit(OP_JUMPI);
            } else if (funct3 == 1) {  /* BNE */
                if (rs2 == 0) {
                    load_reg(rs1);
                    /* Non-zero value is the condition */
                } else if (rs1 == 0) {
                    load_reg(rs2);
                } else {
                    load_reg(rs1);
                    load_reg(rs2);
                    emit(OP_EQ);
                    emit(OP_ISZERO);
                }
                emit(OP_PUSH2);
                add_pending_jump(out_pos, target_pc);
                emit(0); emit(0);
                emit(OP_JUMPI);
            }
        }
        break;

    case 0x6F:  /* JAL */
        {
            int32_t imm = get_imm_j(instr);
            uint32_t target_pc = pc + imm;

            /* Store return address if rd != 0 */
            if (rd != 0) {
                push_u32(pc + 4);
                store_reg(rd);
            }

            /* Jump to target */
            emit(OP_PUSH2);
            add_pending_jump(out_pos, target_pc);
            emit(0); emit(0);
            emit(OP_JUMP);
        }
        break;

    case 0x73:  /* SYSTEM */
        if (instr == 0x00000073) {  /* ECALL */
            /* Return a0 value */
            load_reg(10);  /* a0 */
            push_u32(RETVAL_ADDR);
            emit(OP_MSTORE);
            /* Return 32 bytes from RETVAL_ADDR */
            push1(32);
            push_u32(RETVAL_ADDR);
            emit(OP_RETURN);
        }
        break;
    }
}

/* Emit initialization code */
static void emit_init(uint32_t sp_value) {
    /* Initialize stack pointer (x2) */
    push_u32(sp_value);
    push_u32(REG_BASE + 2 * REG_SIZE);
    emit(OP_MSTORE);
}

/* Main compilation function */
uint32_t compile(uint32_t *input, uint32_t num_instrs, uint8_t *out_buf) {
    output = out_buf;
    out_pos = 0;
    num_pending = 0;

    /* Initialize */
    emit_init(0x80010000);

    /* First pass: compile instructions and record PC mappings */
    for (uint32_t i = 0; i < num_instrs; i++) {
        uint32_t pc = LOAD_ADDRESS + i * 4;
        uint32_t pc_offset = i;

        /* Record EVM position for this PC */
        pc_to_evm[pc_offset] = out_pos;

        /* Emit JUMPDEST for potential jump targets */
        emit(OP_JUMPDEST);

        /* Compile the instruction */
        compile_instruction(pc, input[i]);
    }

    /* Resolve pending jumps */
    for (uint32_t i = 0; i < num_pending; i++) {
        uint32_t jump_pos = pending_jumps[i][0];
        uint32_t target_pc = pending_jumps[i][1];
        uint32_t target_offset = (target_pc - LOAD_ADDRESS) / 4;
        uint32_t evm_target = pc_to_evm[target_offset];

        /* Patch the jump target */
        output[jump_pos] = (evm_target >> 8) & 0xFF;
        output[jump_pos + 1] = evm_target & 0xFF;
    }

    return out_pos;
}

/* Entry point - called from assembly startup */
/* When compiled to rv32im and run on EVM, this reads from fixed memory addresses */
int main(void) {
    /* Read input from memory */
    volatile uint32_t *input_len_ptr = (volatile uint32_t *)INPUT_LEN_ADDR;
    volatile uint32_t *input_ptr = (volatile uint32_t *)INPUT_ADDR;
    volatile uint8_t *output_ptr = (volatile uint8_t *)OUTPUT_ADDR;
    volatile uint32_t *output_len_ptr = (volatile uint32_t *)OUTPUT_LEN_ADDR;

    uint32_t input_len = *input_len_ptr;
    uint32_t num_instrs = input_len / 4;

    /* Compile */
    uint32_t out_len = compile((uint32_t *)input_ptr, num_instrs, (uint8_t *)output_ptr);

    /* Store output length */
    *output_len_ptr = out_len;

    /* Return output length in a0 */
    return (int)out_len;
}
