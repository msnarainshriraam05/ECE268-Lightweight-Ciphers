/*
 * GIFT-128/128 — reference implementation.
 * 128-bit block, 128-bit key, 40 rounds.
 *
 * Ported from the official nibble-based reference:
 *   giftcipher/gift  GIFT128-128_cipher.cpp  (Siang Meng Sim, March 2017)
 *
 * Reference: Banik et al., "GIFT: A Small Present", ePrint 2017/622.
 */

#include "gift128.h"
#include <string.h>

/* ── S-box and inverse ─────────────────────────────────────────── */
static const uint8_t GIFT_S[16] = {
    1, 10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14
};
static const uint8_t GIFT_S_inv[16] = {
    13, 0, 8, 6, 2, 12, 4, 11, 14, 7, 1, 10, 3, 9, 15, 5
};

/* ── 128-bit permutation tables ────────────────────────────────── */
static const uint8_t GIFT_P[128] = {
     0, 33, 66, 99, 96,  1, 34, 67, 64, 97,  2, 35, 32, 65, 98,  3,
     4, 37, 70,103,100,  5, 38, 71, 68,101,  6, 39, 36, 69,102,  7,
     8, 41, 74,107,104,  9, 42, 75, 72,105, 10, 43, 40, 73,106, 11,
    12, 45, 78,111,108, 13, 46, 79, 76,109, 14, 47, 44, 77,110, 15,
    16, 49, 82,115,112, 17, 50, 83, 80,113, 18, 51, 48, 81,114, 19,
    20, 53, 86,119,116, 21, 54, 87, 84,117, 22, 55, 52, 85,118, 23,
    24, 57, 90,123,120, 25, 58, 91, 88,121, 26, 59, 56, 89,122, 27,
    28, 61, 94,127,124, 29, 62, 95, 92,125, 30, 63, 60, 93,126, 31
};

static const uint8_t GIFT_P_inv[128] = {
     0,  5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63,
    64, 69, 74, 79, 80, 85, 90, 95, 96,101,106,111,112,117,122,127,
    12,  1,  6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59,
    76, 65, 70, 75, 92, 81, 86, 91,108, 97,102,107,124,113,118,123,
     8, 13,  2,  7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55,
    72, 77, 66, 71, 88, 93, 82, 87,104,109, 98,103,120,125,114,119,
     4,  9, 14,  3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51,
    68, 73, 78, 67, 84, 89, 94, 83,100,105,110, 99,116,121,126,115
};

/* ── round constants (6-bit LFSR, 40 rounds needed) ────────────── */
static const uint8_t GIFT_RC[40] = {
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B,
    0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
    0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30,
    0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
    0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A
};

/* ──────────────────────────────────────────────────────────────── */
/*  Internal state: 32 nibbles, cells[0] = LSB nibble.             */
/*  Byte[0] is MSB: byte[i] = (cells[31-2i] << 4) | cells[30-2i]  */
/* ──────────────────────────────────────────────────────────────── */

static void bytes_to_nibbles(const uint8_t in[16], uint8_t out[32]) {
    for (int i = 0; i < 16; i++) {
        out[31 - 2 * i]     = in[i] >> 4;       /* high nibble */
        out[31 - 2 * i - 1] = in[i] & 0x0f;     /* low nibble  */
    }
}

static void nibbles_to_bytes(const uint8_t in[32], uint8_t out[16]) {
    for (int i = 0; i < 16; i++)
        out[i] = (in[31 - 2 * i] << 4) | in[30 - 2 * i];
}

/* ── key state: also 32 nibbles ────────────────────────────────── */

static void key_update(uint8_t k[32]) {
    uint8_t tmp[32];
    for (int i = 0; i < 32; i++)
        tmp[i] = k[(i + 8) % 32];

    for (int i = 0; i < 24; i++)
        k[i] = tmp[i];

    /* k_0 >>> 12  (nibble rotation within 16-bit word) */
    k[24] = tmp[27]; k[25] = tmp[24];
    k[26] = tmp[25]; k[27] = tmp[26];

    /* k_1 >>> 2   (2-bit rotation within each nibble pair) */
    k[28] = ((tmp[28] & 0xc) >> 2) ^ ((tmp[29] & 0x3) << 2);
    k[29] = ((tmp[29] & 0xc) >> 2) ^ ((tmp[30] & 0x3) << 2);
    k[30] = ((tmp[30] & 0xc) >> 2) ^ ((tmp[31] & 0x3) << 2);
    k[31] = ((tmp[31] & 0xc) >> 2) ^ ((tmp[28] & 0x3) << 2);
}

/* ── pre-compute round keys as two 32-bit words per round ──────── */
/*   rk[r][0] = lower 32 key bits (added to b1 positions)           */
/*   rk[r][1] = upper 32 key bits (added to b2 positions)           */
void gift128_key_schedule(const uint8_t key[16],
                          uint32_t rk[GIFT128_ROUNDS][2]) {
    uint8_t kstate[32];
    bytes_to_nibbles(key, kstate);

    for (int r = 0; r < GIFT128_ROUNDS; r++) {
        /* extract bits 0..31 and bits 64..95 from key nibbles */
        uint8_t kbits[128];
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                kbits[4 * i + j] = (kstate[i] >> j) & 1;

        rk[r][0] = rk[r][1] = 0;
        for (int b = 0; b < 32; b++) {
            rk[r][0] |= (uint32_t)kbits[b]      << b;
            rk[r][1] |= (uint32_t)kbits[b + 64] << b;
        }

        key_update(kstate);
    }
}

/* ── SubCells ──────────────────────────────────────────────────── */
static void subcells(uint8_t s[32]) {
    for (int i = 0; i < 32; i++) s[i] = GIFT_S[s[i]];
}
static void subcells_inv(uint8_t s[32]) {
    for (int i = 0; i < 32; i++) s[i] = GIFT_S_inv[s[i]];
}

/* ── PermBits ──────────────────────────────────────────────────── */
static void permbits(uint8_t s[32]) {
    uint8_t bits[128], out[128];
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 4; j++)
            bits[4 * i + j] = (s[i] >> j) & 1;
    for (int i = 0; i < 128; i++)
        out[GIFT_P[i]] = bits[i];
    for (int i = 0; i < 32; i++) {
        s[i] = 0;
        for (int j = 0; j < 4; j++)
            s[i] |= out[4 * i + j] << j;
    }
}
static void permbits_inv(uint8_t s[32]) {
    uint8_t bits[128], out[128];
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 4; j++)
            bits[4 * i + j] = (s[i] >> j) & 1;
    for (int i = 0; i < 128; i++)
        out[GIFT_P_inv[i]] = bits[i];
    for (int i = 0; i < 32; i++) {
        s[i] = 0;
        for (int j = 0; j < 4; j++)
            s[i] |= out[4 * i + j] << j;
    }
}

/* ── AddRoundKey ───────────────────────────────────────────────── */
static void add_round_key(uint8_t s[32], const uint32_t rk[2], int round) {
    uint8_t bits[128];
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 4; j++)
            bits[4 * i + j] = (s[i] >> j) & 1;

    for (int i = 0; i < 32; i++) {
        bits[4 * i + 1] ^= (rk[0] >> i) & 1;
        bits[4 * i + 2] ^= (rk[1] >> i) & 1;
    }

    bits[3]   ^=  GIFT_RC[round]       & 1;
    bits[7]   ^= (GIFT_RC[round] >> 1) & 1;
    bits[11]  ^= (GIFT_RC[round] >> 2) & 1;
    bits[15]  ^= (GIFT_RC[round] >> 3) & 1;
    bits[19]  ^= (GIFT_RC[round] >> 4) & 1;
    bits[23]  ^= (GIFT_RC[round] >> 5) & 1;
    bits[127] ^= 1;

    for (int i = 0; i < 32; i++) {
        s[i] = 0;
        for (int j = 0; j < 4; j++)
            s[i] |= bits[4 * i + j] << j;
    }
}

/* ── encrypt ───────────────────────────────────────────────────── */
void gift128_encrypt(const uint32_t rk[GIFT128_ROUNDS][2],
                     const uint8_t pt[16], uint8_t ct[16]) {
    uint8_t s[32];
    bytes_to_nibbles(pt, s);

    for (int r = 0; r < GIFT128_ROUNDS; r++) {
        subcells(s);
        permbits(s);
        add_round_key(s, rk[r], r);
    }

    nibbles_to_bytes(s, ct);
}

/* ── decrypt ───────────────────────────────────────────────────── */
void gift128_decrypt(const uint32_t rk[GIFT128_ROUNDS][2],
                     const uint8_t ct[16], uint8_t pt[16]) {
    uint8_t s[32];
    bytes_to_nibbles(ct, s);

    for (int r = GIFT128_ROUNDS - 1; r >= 0; r--) {
        add_round_key(s, rk[r], r);
        permbits_inv(s);
        subcells_inv(s);
    }

    nibbles_to_bytes(s, pt);
}
