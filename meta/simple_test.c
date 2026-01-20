/*
 * Simple test: read a value from memory and return it
 */

#include <stdint.h>

#define INPUT_ADDR 0x80000C00

int main(void) {
    volatile uint32_t *input = (volatile uint32_t *)INPUT_ADDR;
    return (int)(*input);
}
