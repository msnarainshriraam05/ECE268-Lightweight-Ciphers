/*
 * test_vectors.c — validation suite for SIMON 64/128, GIFT-128/128,
 *                  and CBC / CTR mode wrappers.
 *
 * Build:  gcc -O2 -Wall -o test_vectors test_vectors.c simon64.c gift128.c modes.c
 * Run:    ./test_vectors
 *
 * Test vectors sourced from:
 *   SIMON — ePrint 2013/404 Appendix B (SIMON 64/128)
 *   GIFT  — ePrint 2017/622 Section 5  (GIFT-128/128)
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "simon64.h"
#include "gift128.h"
#include "modes.h"

/* ── helpers ───────────────────────────────────────────────────── */
static int total = 0, passed = 0;

static void hex_dump(const uint8_t *buf, size_t len) {
    for (size_t i = 0; i < len; i++) printf("%02x", buf[i]);
}

static int check(const char *label, const uint8_t *got, const uint8_t *want, size_t n) {
    total++;
    if (memcmp(got, want, n) == 0) {
        printf("  [PASS] %s\n", label);
        passed++;
        return 1;
    }
    printf("  [FAIL] %s\n         got  ", label);
    hex_dump(got, n);
    printf("\n         want ");
    hex_dump(want, n);
    printf("\n");
    return 0;
}

/* ── wrappers so block ciphers match the block_fn_t signature ── */

typedef struct { uint32_t rk[SIMON64_ROUNDS]; } Simon64Ctx;
typedef struct { uint32_t rk[GIFT128_ROUNDS][2]; } Gift128Ctx;

static void simon64_enc_wrap(const void *ctx, const uint8_t *in, uint8_t *out) {
    simon64_encrypt(((const Simon64Ctx *)ctx)->rk, in, out);
}
static void simon64_dec_wrap(const void *ctx, const uint8_t *in, uint8_t *out) {
    simon64_decrypt(((const Simon64Ctx *)ctx)->rk, in, out);
}
static void gift128_enc_wrap(const void *ctx, const uint8_t *in, uint8_t *out) {
    gift128_encrypt(((const Gift128Ctx *)ctx)->rk, in, out);
}
static void gift128_dec_wrap(const void *ctx, const uint8_t *in, uint8_t *out) {
    gift128_decrypt(((const Gift128Ctx *)ctx)->rk, in, out);
}

/* ================================================================
 *  1.  SIMON 64/128  —  ECB test vector from ePrint 2013/404
 * ================================================================ */
static void test_simon64_ecb(void) {
    printf("\n=== SIMON 64/128  ECB ===\n");

    /* Key  K = 1b1a1918 13121110 0b0a0908 03020100
     *        k_0 = 0x03020100, k_3 = 0x1b1a1918 */
    const uint32_t key[4] = { 0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918 };
    const uint8_t  pt[8]  = { 0x65,0x6b,0x69,0x6c, 0x20,0x64,0x6e,0x75 };
    const uint8_t  want[8]= { 0x44,0xc8,0xfc,0x20, 0xb9,0xdf,0xa0,0x7a };

    uint32_t rk[SIMON64_ROUNDS];
    simon64_key_schedule(key, rk);

    uint8_t ct[8], rt[8];
    simon64_encrypt(rk, pt, ct);
    check("encrypt", ct, want, 8);

    simon64_decrypt(rk, ct, rt);
    check("decrypt round-trip", rt, pt, 8);
}

/* ================================================================
 *  2.  GIFT-128/128  —  ECB test vectors from ePrint 2017/622
 * ================================================================ */
static void test_gift128_ecb(void) {
    printf("\n=== GIFT-128/128  ECB ===\n");

    /* Vector 1: all-zero key and plaintext */
    {
        const uint8_t key[16] = {0};
        const uint8_t pt[16]  = {0};
        const uint8_t want[16] = {
            0xcd,0x0b,0xd7,0x38, 0x38,0x8a,0xd3,0xf6,
            0x68,0xb1,0x5a,0x36, 0xce,0xb6,0xff,0x92
        };

        uint32_t rk[GIFT128_ROUNDS][2];
        gift128_key_schedule(key, rk);

        uint8_t ct[16], rt[16];
        gift128_encrypt(rk, pt, ct);
        check("v1 encrypt (zero key)", ct, want, 16);

        gift128_decrypt(rk, ct, rt);
        check("v1 decrypt round-trip", rt, pt, 16);
    }

    /* Vector 2: key = pt = fedcba9876543210 fedcba9876543210 */
    {
        const uint8_t key[16] = {
            0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10,
            0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10
        };
        const uint8_t pt[16] = {
            0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10,
            0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10
        };
        const uint8_t want[16] = {
            0x84,0x22,0x24,0x1a, 0x6d,0xbf,0x5a,0x93,
            0x46,0xaf,0x46,0x84, 0x09,0xee,0x01,0x52
        };

        uint32_t rk[GIFT128_ROUNDS][2];
        gift128_key_schedule(key, rk);

        uint8_t ct[16], rt[16];
        gift128_encrypt(rk, pt, ct);
        check("v2 encrypt (fedc... key)", ct, want, 16);

        gift128_decrypt(rk, ct, rt);
        check("v2 decrypt round-trip", rt, pt, 16);
    }
}

/* ================================================================
 *  3.  SIMON 64/128 — CBC mode (encrypt + decrypt round-trip)
 * ================================================================ */
static void test_simon64_cbc(void) {
    printf("\n=== SIMON 64/128  CBC ===\n");

    const uint32_t key[4] = { 0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918 };
    Simon64Ctx ctx;
    simon64_key_schedule(key, ctx.rk);

    const uint8_t iv[8] = { 0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08 };
    const uint8_t pt[] = "Hello, lightweight crypto!";  /* 26 bytes */
    const size_t pt_len = 26;

    uint8_t ct[40], rt[40];
    size_t ct_len = cbc_encrypt(simon64_enc_wrap, &ctx, 8, iv, pt, pt_len, ct);

    /* ct_len must be a multiple of block size and > pt_len */
    total++;
    if (ct_len == 32) { printf("  [PASS] CBC enc length = 32\n"); passed++; }
    else              { printf("  [FAIL] CBC enc length = %zu (want 32)\n", ct_len); }

    size_t rt_len = cbc_decrypt(simon64_dec_wrap, &ctx, 8, iv, ct, ct_len, rt);

    total++;
    if (rt_len == pt_len && memcmp(rt, pt, pt_len) == 0) {
        printf("  [PASS] CBC decrypt round-trip\n"); passed++;
    } else {
        printf("  [FAIL] CBC decrypt round-trip (got len %zu)\n", rt_len);
    }
}

/* ================================================================
 *  4.  GIFT-128/128 — CBC mode
 * ================================================================ */
static void test_gift128_cbc(void) {
    printf("\n=== GIFT-128/128  CBC ===\n");

    const uint8_t key[16] = {
        0x00,0x11,0x22,0x33, 0x44,0x55,0x66,0x77,
        0x88,0x99,0xaa,0xbb, 0xcc,0xdd,0xee,0xff
    };
    Gift128Ctx ctx;
    gift128_key_schedule(key, ctx.rk);

    const uint8_t iv[16] = {
        0xf0,0xe0,0xd0,0xc0, 0xb0,0xa0,0x90,0x80,
        0x70,0x60,0x50,0x40, 0x30,0x20,0x10,0x00
    };
    const uint8_t pt[] = "GIFT-128 CBC mode test!!!";  /* 25 bytes */
    const size_t pt_len = 25;

    uint8_t ct[48], rt[48];
    size_t ct_len = cbc_encrypt(gift128_enc_wrap, &ctx, 16, iv, pt, pt_len, ct);

    total++;
    if (ct_len == 32) { printf("  [PASS] CBC enc length = 32\n"); passed++; }
    else              { printf("  [FAIL] CBC enc length = %zu (want 32)\n", ct_len); }

    size_t rt_len = cbc_decrypt(gift128_dec_wrap, &ctx, 16, iv, ct, ct_len, rt);

    total++;
    if (rt_len == pt_len && memcmp(rt, pt, pt_len) == 0) {
        printf("  [PASS] CBC decrypt round-trip\n"); passed++;
    } else {
        printf("  [FAIL] CBC decrypt round-trip (got len %zu)\n", rt_len);
    }
}

/* ================================================================
 *  5.  SIMON 64/128 — CTR mode (symmetric; encrypt == decrypt)
 * ================================================================ */
static void test_simon64_ctr(void) {
    printf("\n=== SIMON 64/128  CTR ===\n");

    const uint32_t key[4] = { 0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918 };
    Simon64Ctx ctx;
    simon64_key_schedule(key, ctx.rk);

    const uint8_t nonce[8] = { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x01 };
    const uint8_t pt[] = "CTR mode needs no padding";  /* 25 bytes */
    const size_t  pt_len = 25;

    uint8_t ct[32], rt[32];
    ctr_crypt(simon64_enc_wrap, &ctx, 8, nonce, pt, pt_len, ct);
    ctr_crypt(simon64_enc_wrap, &ctx, 8, nonce, ct, pt_len, rt);

    check("CTR round-trip", rt, pt, pt_len);
}

/* ================================================================
 *  6.  GIFT-128/128 — CTR mode
 * ================================================================ */
static void test_gift128_ctr(void) {
    printf("\n=== GIFT-128/128  CTR ===\n");

    const uint8_t key[16] = {
        0xde,0xad,0xbe,0xef, 0xca,0xfe,0xba,0xbe,
        0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef
    };
    Gift128Ctx ctx;
    gift128_key_schedule(key, ctx.rk);

    const uint8_t nonce[16] = {0};
    const uint8_t pt[] = "CTR: no padding, arbitrary length message!";  /* 43 bytes */
    const size_t  pt_len = 43;

    uint8_t ct[48], rt[48];
    ctr_crypt(gift128_enc_wrap, &ctx, 16, nonce, pt, pt_len, ct);
    ctr_crypt(gift128_enc_wrap, &ctx, 16, nonce, ct, pt_len, rt);

    check("CTR round-trip", rt, pt, pt_len);
}

/* ================================================================
 *  7.  PKCS#7 padding edge cases
 * ================================================================ */
static void test_pkcs7_edge(void) {
    printf("\n=== PKCS#7 padding edge cases ===\n");

    const uint32_t key[4] = { 0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918 };
    Simon64Ctx ctx;
    simon64_key_schedule(key, ctx.rk);
    const uint8_t iv[8] = {0};

    /* 8-byte plaintext (exactly one block → full padding block appended) */
    {
        const uint8_t pt[8] = { 0xAA,0xBB,0xCC,0xDD, 0xEE,0xFF,0x00,0x11 };
        uint8_t ct[24], rt[24];

        size_t ct_len = cbc_encrypt(simon64_enc_wrap, &ctx, 8, iv, pt, 8, ct);
        total++;
        if (ct_len == 16) { printf("  [PASS] 1-block pad length = 16\n"); passed++; }
        else              { printf("  [FAIL] 1-block pad length = %zu\n", ct_len); }

        size_t rt_len = cbc_decrypt(simon64_dec_wrap, &ctx, 8, iv, ct, ct_len, rt);
        total++;
        if (rt_len == 8 && memcmp(rt, pt, 8) == 0) {
            printf("  [PASS] 1-block pad round-trip\n"); passed++;
        } else {
            printf("  [FAIL] 1-block pad round-trip (len %zu)\n", rt_len);
        }
    }

    /* 1-byte plaintext */
    {
        const uint8_t pt[1] = { 0x42 };
        uint8_t ct[16], rt[16];

        size_t ct_len = cbc_encrypt(simon64_enc_wrap, &ctx, 8, iv, pt, 1, ct);
        total++;
        if (ct_len == 8) { printf("  [PASS] 1-byte pad length = 8\n"); passed++; }
        else             { printf("  [FAIL] 1-byte pad length = %zu\n", ct_len); }

        size_t rt_len = cbc_decrypt(simon64_dec_wrap, &ctx, 8, iv, ct, ct_len, rt);
        total++;
        if (rt_len == 1 && rt[0] == 0x42) {
            printf("  [PASS] 1-byte pad round-trip\n"); passed++;
        } else {
            printf("  [FAIL] 1-byte pad round-trip (len %zu)\n", rt_len);
        }
    }
}

/* ================================================================ */
int main(void) {
    test_simon64_ecb();
    test_gift128_ecb();
    test_simon64_cbc();
    test_gift128_cbc();
    test_simon64_ctr();
    test_gift128_ctr();
    test_pkcs7_edge();

    printf("\n────────────────────────────────\n");
    printf("  %d / %d tests passed\n", passed, total);
    printf("────────────────────────────────\n");

    return (passed == total) ? 0 : 1;
}
 
 