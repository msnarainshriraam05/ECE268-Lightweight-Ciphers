/*
 * SIMON 64/128 — reference implementation.
 * 64-bit block (2×32-bit words), 128-bit key (4×32-bit words), 44 rounds.
 *
 * Key schedule uses z3 sequence; m = 4.
 * Byte I/O is big-endian to match the spec's hex notation.
 *
 * Reference: Beaulieu et al., "The SIMON and SPECK Families of
 *            Lightweight Block Ciphers", ePrint 2013/404.
 */

#include "simon64.h"

/* ── helpers ───────────────────────────────────────────────────── */
static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}
static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* f(x) = (x<<<1 & x<<<8) ^ x<<<2 */
static inline uint32_t simon_f(uint32_t x) {
    return (rotl32(x, 1) & rotl32(x, 8)) ^ rotl32(x, 2);
}

/* z3 constant for SIMON 64/128 (n=32, m=4) — from ePrint 2013/404 Table 4 */
static const uint64_t z3 = UINT64_C(0xfc2ce51207a635db);

/* ── key schedule ──────────────────────────────────────────────── */
/*
 * key[0..3] = k_0, k_1, k_2, k_3 (least-significant word first).
 */
void simon64_key_schedule(const uint32_t key[4], uint32_t rk[SIMON64_ROUNDS]) {
    rk[0] = key[0];
    rk[1] = key[1];
    rk[2] = key[2];
    rk[3] = key[3];

    for (int i = 4; i < SIMON64_ROUNDS; i++) {
        uint32_t tmp = rotr32(rk[i - 1], 3) ^ rk[i - 3];
        tmp ^= rotr32(tmp, 1);
        rk[i] = ~rk[i - 4] ^ tmp
              ^ (uint32_t)((z3 >> ((i - 4) % 62)) & 1) ^ 3u;
    }
}

/* ── big-endian block load / store ─────────────────────────────── */
static inline void load_be(const uint8_t b[8], uint32_t *x, uint32_t *y) {
    *x = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
         ((uint32_t)b[2] <<  8) |  (uint32_t)b[3];
    *y = ((uint32_t)b[4] << 24) | ((uint32_t)b[5] << 16) |
         ((uint32_t)b[6] <<  8) |  (uint32_t)b[7];
}

static inline void store_be(uint8_t b[8], uint32_t x, uint32_t y) {
    b[0] = x >> 24; b[1] = x >> 16; b[2] = x >> 8; b[3] = x;
    b[4] = y >> 24; b[5] = y >> 16; b[6] = y >> 8; b[7] = y;
}

/* ── encrypt: R_k(x,y) = (y ^ f(x) ^ k, x) ───────────────────── */
void simon64_encrypt(const uint32_t rk[SIMON64_ROUNDS],
                     const uint8_t pt[8], uint8_t ct[8]) {
    uint32_t x, y;
    load_be(pt, &x, &y);

    for (int i = 0; i < SIMON64_ROUNDS; i++) {
        uint32_t tmp = x;
        x = y ^ simon_f(x) ^ rk[i];
        y = tmp;
    }

    store_be(ct, x, y);
}

/* ── decrypt ───────────────────────────────────────────────────── */
void simon64_decrypt(const uint32_t rk[SIMON64_ROUNDS],
                     const uint8_t ct[8], uint8_t pt[8]) {
    uint32_t x, y;
    load_be(ct, &x, &y);

    for (int i = SIMON64_ROUNDS - 1; i >= 0; i--) {
        uint32_t tmp = y;
        y = x ^ simon_f(y) ^ rk[i];
        x = tmp;
    }

    store_be(pt, x, y);
}
