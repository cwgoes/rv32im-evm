# Makefile for building SHA256 benchmark for rv32im
#
# Requires riscv32 toolchain. Common names:
# - riscv32-unknown-elf-gcc (standard)
# - riscv64-unknown-elf-gcc with -march=rv32im (also works)
# - riscv-none-embed-gcc (embedded toolchain)

# Try to find a working RISC-V compiler
CROSS_COMPILE ?= $(shell which riscv32-unknown-elf-gcc > /dev/null 2>&1 && echo riscv32-unknown-elf- || \
                         (which riscv64-unknown-elf-gcc > /dev/null 2>&1 && echo riscv64-unknown-elf- || \
                         (which riscv-none-embed-gcc > /dev/null 2>&1 && echo riscv-none-embed- || echo "")))

CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump

# Compiler flags for rv32im (no atomic, no FP, no compressed)
CFLAGS = -march=rv32im -mabi=ilp32 -O2 -ffreestanding -nostdlib -nostartfiles
CFLAGS += -fno-builtin -fno-stack-protector -fno-pie -fno-pic
CFLAGS += -Wall -Wextra

ASFLAGS = -march=rv32im -mabi=ilp32

LDFLAGS = -T link.ld -nostdlib -static

# Source files
SRC_DIR = src
SRCS_C = $(SRC_DIR)/sha256.c
SRCS_S = $(SRC_DIR)/start.S
OBJS = $(SRCS_C:.c=.o) $(SRCS_S:.S=.o)

# Output
TARGET = sha256
ELF = $(TARGET).elf
BIN = $(TARGET).bin
HEX = $(TARGET).hex
DUMP = $(TARGET).dump

.PHONY: all clean check-compiler

all: check-compiler $(BIN) $(HEX) $(DUMP)

check-compiler:
ifeq ($(CROSS_COMPILE),)
	@echo "ERROR: No RISC-V cross-compiler found!"
	@echo "Please install one of:"
	@echo "  - riscv32-unknown-elf-gcc"
	@echo "  - riscv64-unknown-elf-gcc"
	@echo "  - riscv-none-embed-gcc"
	@echo ""
	@echo "On Ubuntu/Debian: sudo apt install gcc-riscv64-unknown-elf"
	@echo "Or use: https://github.com/riscv-collab/riscv-gnu-toolchain"
	@exit 1
endif
	@echo "Using compiler: $(CC)"

$(ELF): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^
	@echo "Built $(ELF)"

$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@
	@echo "Created raw binary: $(BIN) ($(shell wc -c < $@) bytes)"

$(HEX): $(ELF)
	$(OBJCOPY) -O ihex $< $@

$(DUMP): $(ELF)
	$(OBJDUMP) -d -S $< > $@
	@echo "Created disassembly: $(DUMP)"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S
	$(CC) $(ASFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(ELF) $(BIN) $(HEX) $(DUMP)

# Show info about the binary
info: $(ELF)
	@echo "=== Binary Info ==="
	$(CROSS_COMPILE)size $(ELF)
	@echo ""
	@echo "=== Sections ==="
	$(CROSS_COMPILE)readelf -S $(ELF)
