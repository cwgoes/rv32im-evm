/**
 * Standalone SHA-256 implementation for bare-metal RISC-V
 * No libc dependencies - suitable for rv32im compilation
 */

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;

/* SHA-256 constants - first 32 bits of fractional parts of cube roots of first 64 primes */
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Initial hash values - first 32 bits of fractional parts of square roots of first 8 primes */
static const uint32_t H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* Rotate right */
static inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* SHA-256 functions */
static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static inline uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static inline uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

/* Process a 64-byte block */
static void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;
    int i;

    /* Prepare message schedule */
    for (i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }
    for (i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    /* Initialize working variables */
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    /* Main loop */
    for (i = 0; i < 64; i++) {
        t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    /* Add compressed chunk to current hash value */
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/* SHA-256 context */
typedef struct {
    uint32_t state[8];
    uint64_t bitcount;
    uint8_t buffer[64];
} sha256_ctx;

/* Initialize SHA-256 context */
void sha256_init(sha256_ctx *ctx) {
    int i;
    for (i = 0; i < 8; i++) {
        ctx->state[i] = H0[i];
    }
    ctx->bitcount = 0;
    for (i = 0; i < 64; i++) {
        ctx->buffer[i] = 0;
    }
}

/* Update SHA-256 context with data */
void sha256_update(sha256_ctx *ctx, const uint8_t *data, uint32_t len) {
    uint32_t i;
    uint32_t index = (ctx->bitcount >> 3) & 0x3F;  /* Position in buffer */

    ctx->bitcount += ((uint64_t)len << 3);

    for (i = 0; i < len; i++) {
        ctx->buffer[index++] = data[i];
        if (index == 64) {
            sha256_transform(ctx->state, ctx->buffer);
            index = 0;
        }
    }
}

/* Finalize SHA-256 and produce hash */
void sha256_final(sha256_ctx *ctx, uint8_t hash[32]) {
    uint32_t index = (ctx->bitcount >> 3) & 0x3F;
    int i;

    /* Pad message */
    ctx->buffer[index++] = 0x80;
    if (index > 56) {
        while (index < 64) {
            ctx->buffer[index++] = 0;
        }
        sha256_transform(ctx->state, ctx->buffer);
        index = 0;
    }
    while (index < 56) {
        ctx->buffer[index++] = 0;
    }

    /* Append length in bits (big-endian) */
    ctx->buffer[56] = (ctx->bitcount >> 56) & 0xFF;
    ctx->buffer[57] = (ctx->bitcount >> 48) & 0xFF;
    ctx->buffer[58] = (ctx->bitcount >> 40) & 0xFF;
    ctx->buffer[59] = (ctx->bitcount >> 32) & 0xFF;
    ctx->buffer[60] = (ctx->bitcount >> 24) & 0xFF;
    ctx->buffer[61] = (ctx->bitcount >> 16) & 0xFF;
    ctx->buffer[62] = (ctx->bitcount >> 8) & 0xFF;
    ctx->buffer[63] = ctx->bitcount & 0xFF;

    sha256_transform(ctx->state, ctx->buffer);

    /* Produce final hash (big-endian) */
    for (i = 0; i < 8; i++) {
        hash[i*4] = (ctx->state[i] >> 24) & 0xFF;
        hash[i*4+1] = (ctx->state[i] >> 16) & 0xFF;
        hash[i*4+2] = (ctx->state[i] >> 8) & 0xFF;
        hash[i*4+3] = ctx->state[i] & 0xFF;
    }
}

/* Simple SHA-256 of a single message */
void sha256(const uint8_t *data, uint32_t len, uint8_t hash[32]) {
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, hash);
}

/* Entry point - compute SHA256 of a test message and return first 4 bytes as result */
/* Message is passed via memory at address 0x80020000, length at 0x80020100 */
volatile uint8_t *INPUT_DATA = (volatile uint8_t *)0x80020000;
volatile uint32_t *INPUT_LEN = (volatile uint32_t *)0x80020100;
volatile uint8_t *OUTPUT_HASH = (volatile uint8_t *)0x80020200;

int main(void) {
    uint8_t hash[32];
    uint32_t len = *INPUT_LEN;
    int i;

    /* Compute SHA-256 */
    sha256((const uint8_t *)INPUT_DATA, len, hash);

    /* Store result */
    for (i = 0; i < 32; i++) {
        OUTPUT_HASH[i] = hash[i];
    }

    /* Return first 4 bytes of hash as a single value for easy verification */
    return ((uint32_t)hash[0] << 24) | ((uint32_t)hash[1] << 16) |
           ((uint32_t)hash[2] << 8) | (uint32_t)hash[3];
}
