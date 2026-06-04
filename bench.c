/*
 * bench.c — benchmark SIMON 64/128, GIFT-128, and AES-128 baseline.
 *
 * Measures:
 *   1. Key-schedule cost (ns per call, averaged over many iterations)
 *   2. ECB throughput    (cycles/byte, estimated from wall-clock time)
 *   3. CTR throughput    (cycles/byte)
 *
 * AES-128 baseline uses CommonCrypto (macOS) or OpenSSL (Linux).
 *
 * Build:  gcc -O2 -Wall -o bench bench.c simon64.c gift128.c modes.c -framework Security
 *         (the Makefile handles the correct flags)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "simon64.h"
#include "gift128.h"
#include "modes.h"

/* ── portable high-resolution timer ────────────────────────────── */
#ifdef __APPLE__
#include <mach/mach_time.h>
static double ns_per_tick = 0.0;
static void timer_init(void) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    ns_per_tick = (double)tb.numer / (double)tb.denom;
}
static uint64_t timer_ns(void) {
    return (uint64_t)(mach_absolute_time() * ns_per_tick);
}
#else
static void timer_init(void) { (void)0; }
static uint64_t timer_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

/* ── AES-128 baseline via CommonCrypto (macOS) ─────────────────── */
#ifdef __APPLE__
#include <CommonCrypto/CommonCryptor.h>

typedef struct { CCCryptorRef enc; } AES128Ctx;

static void aes128_setup(AES128Ctx *ctx, const uint8_t key[16]) {
    CCCryptorCreate(kCCEncrypt, kCCAlgorithmAES128, kCCOptionECBMode,
                    key, 16, NULL, &ctx->enc);
}
static void aes128_encrypt_block(const void *vctx, const uint8_t *in, uint8_t *out) {
    size_t moved = 0;
    CCCryptorUpdate(((const AES128Ctx *)vctx)->enc, in, 16, out, 16, &moved);
}
static void aes128_teardown(AES128Ctx *ctx) {
    CCCryptorRelease(ctx->enc);
}
#define HAS_AES_BASELINE 1
#else
#define HAS_AES_BASELINE 0
#endif

/* ── benchmark parameters ──────────────────────────────────────── */
#define DATA_SIZE   (64 * 1024)   /* 64 KiB */
#define KS_ITERS    100000

/* ── wrappers matching block_fn_t ──────────────────────────────── */
typedef struct { uint32_t rk[SIMON64_ROUNDS]; } S64Ctx;
typedef struct { uint32_t rk[GIFT128_ROUNDS][2]; } G128Ctx;

static void s64_enc(const void *c, const uint8_t *in, uint8_t *out) {
    simon64_encrypt(((const S64Ctx *)c)->rk, in, out);
}
static void g128_enc(const void *c, const uint8_t *in, uint8_t *out) {
    gift128_encrypt(((const G128Ctx *)c)->rk, in, out);
}

/* ── estimate CPU frequency (rough) ────────────────────────────── */
static double estimate_ghz(void) {
    volatile uint64_t dummy = 0;
    uint64_t t0 = timer_ns();
    for (int i = 0; i < 100000000; i++) dummy += i;
    uint64_t t1 = timer_ns();
    (void)dummy;
    double elapsed_s = (t1 - t0) / 1e9;
    /* Very rough: modern CPUs do ~1-4 simple ops/cycle.
       Use 1 GHz as a conservative estimate if we can't determine. */
    return (elapsed_s > 0.001) ? (0.1 / elapsed_s) * 1.0 : 1.0;
}

int main(void) {
    timer_init();

    uint8_t *data_in  = malloc(DATA_SIZE);
    uint8_t *data_out = malloc(DATA_SIZE + 16);
    if (!data_in || !data_out) { perror("malloc"); return 1; }

    /* Fill with pseudo-random data */
    for (size_t i = 0; i < DATA_SIZE; i++) data_in[i] = (uint8_t)(i * 0x9d + 0x37);

    double ghz = estimate_ghz();

    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║          Lightweight Cipher Benchmark (64 KiB)              ║\n");
    printf("╠══════════════════╦══════════╦═════════╦═════════╦═══════════╣\n");
    printf("║ Cipher           ║ Key Sched║ ECB     ║ CTR     ║ Block     ║\n");
    printf("╠══════════════════╬══════════╬═════════╬═════════╬═══════════╣\n");

    /* ── SIMON 64/128 ──────────────────────────────────────── */
    {
        const uint32_t key[4] = { 0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918 };
        S64Ctx ctx;

        uint64_t t0 = timer_ns();
        for (int i = 0; i < KS_ITERS; i++)
            simon64_key_schedule(key, ctx.rk);
        double ks_ns = (double)(timer_ns() - t0) / KS_ITERS;

        /* ECB throughput */
        simon64_key_schedule(key, ctx.rk);
        t0 = timer_ns();
        for (size_t off = 0; off < DATA_SIZE; off += 8)
            simon64_encrypt(ctx.rk, data_in + off, data_out + off);
        double ecb_ns = (double)(timer_ns() - t0);

        /* CTR throughput */
        uint8_t nonce[8] = {0};
        t0 = timer_ns();
        ctr_crypt(s64_enc, &ctx, 8, nonce, data_in, DATA_SIZE, data_out);
        double ctr_ns = (double)(timer_ns() - t0);

        printf("║ %-16s ║ %7.0f  ║ %5.1f   ║ %5.1f   ║ %3d B     ║\n",
               "SIMON 64/128", ks_ns,
               (ecb_ns * ghz) / DATA_SIZE,
               (ctr_ns * ghz) / DATA_SIZE, 8);
    }

    /* ── GIFT-128/128 ──────────────────────────────────────── */
    {
        const uint8_t key[16] = {0};
        G128Ctx ctx;

        uint64_t t0 = timer_ns();
        for (int i = 0; i < KS_ITERS; i++)
            gift128_key_schedule(key, ctx.rk);
        double ks_ns = (double)(timer_ns() - t0) / KS_ITERS;

        /* ECB */
        gift128_key_schedule(key, ctx.rk);
        t0 = timer_ns();
        for (size_t off = 0; off < DATA_SIZE; off += 16)
            gift128_encrypt(ctx.rk, data_in + off, data_out + off);
        double ecb_ns = (double)(timer_ns() - t0);

        /* CTR */
        uint8_t nonce[16] = {0};
        t0 = timer_ns();
        ctr_crypt(g128_enc, &ctx, 16, nonce, data_in, DATA_SIZE, data_out);
        double ctr_ns = (double)(timer_ns() - t0);

        printf("║ %-16s ║ %7.0f  ║ %5.1f   ║ %5.1f   ║ %3d B     ║\n",
               "GIFT-128", ks_ns,
               (ecb_ns * ghz) / DATA_SIZE,
               (ctr_ns * ghz) / DATA_SIZE, 16);
    }

    /* ── AES-128 baseline ──────────────────────────────────── */
#if HAS_AES_BASELINE
    {
        const uint8_t key[16] = {0};
        AES128Ctx ctx;

        uint64_t t0 = timer_ns();
        for (int i = 0; i < KS_ITERS; i++)
            aes128_setup(&ctx, key);
        double ks_ns = (double)(timer_ns() - t0) / KS_ITERS;
        aes128_teardown(&ctx);

        /* ECB */
        aes128_setup(&ctx, key);
        t0 = timer_ns();
        for (size_t off = 0; off < DATA_SIZE; off += 16)
            aes128_encrypt_block(&ctx, data_in + off, data_out + off);
        double ecb_ns = (double)(timer_ns() - t0);

        /* CTR */
        uint8_t nonce[16] = {0};
        t0 = timer_ns();
        ctr_crypt(aes128_encrypt_block, &ctx, 16, nonce,
                  data_in, DATA_SIZE, data_out);
        double ctr_ns = (double)(timer_ns() - t0);

        printf("║ %-16s ║ %7.0f  ║ %5.1f   ║ %5.1f   ║ %3d B     ║\n",
               "AES-128 (hw)", ks_ns,
               (ecb_ns * ghz) / DATA_SIZE,
               (ctr_ns * ghz) / DATA_SIZE, 16);
        aes128_teardown(&ctx);
    }
#endif

    printf("╚══════════════════╩══════════╩═════════╩═════════╩═══════════╝\n");
    printf("\n  cpb = estimated cycles/byte (wall-clock × %.2f GHz)\n", ghz);
    printf("  Key schedule cost averaged over %d iterations.\n", KS_ITERS);
    printf("  Data size: %d KiB.\n\n", DATA_SIZE / 1024);

    /* ── Code size report ──────────────────────────────────── */
    printf("=== Code size (run 'make codesize' for .text segment sizes) ===\n\n");

    free(data_in);
    free(data_out);
    return 0;
}
