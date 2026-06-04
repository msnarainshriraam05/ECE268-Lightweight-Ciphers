#include "modes.h"
#include <string.h>
#include <stdlib.h>

/* ── CBC encrypt ───────────────────────────────────────────────── */
size_t cbc_encrypt(block_fn_t enc, const void *ctx, size_t bs,
                   const uint8_t *iv,
                   const uint8_t *in, size_t len, uint8_t *out) {
    uint8_t prev[64];
    memcpy(prev, iv, bs);

    size_t offset = 0, written = 0;

    /* Full blocks */
    while (offset + bs <= len) {
        uint8_t tmp[64];
        for (size_t i = 0; i < bs; i++) tmp[i] = in[offset + i] ^ prev[i];
        enc(ctx, tmp, out + written);
        memcpy(prev, out + written, bs);
        offset  += bs;
        written += bs;
    }

    /* PKCS#7 padding block */
    uint8_t pad_block[64];
    size_t  rem = len - offset;
    uint8_t pad = (uint8_t)(bs - rem);
    memcpy(pad_block, in + offset, rem);
    memset(pad_block + rem, pad, pad);
    for (size_t i = 0; i < bs; i++) pad_block[i] ^= prev[i];
    enc(ctx, pad_block, out + written);
    written += bs;

    return written;
}

/* ── CBC decrypt ───────────────────────────────────────────────── */
size_t cbc_decrypt(block_fn_t dec, const void *ctx, size_t bs,
                   const uint8_t *iv,
                   const uint8_t *in, size_t len, uint8_t *out) {
    if (len == 0 || len % bs != 0) return (size_t)-1;

    uint8_t prev[64], tmp[64];
    memcpy(prev, iv, bs);

    for (size_t offset = 0; offset < len; offset += bs) {
        dec(ctx, in + offset, tmp);
        for (size_t i = 0; i < bs; i++) out[offset + i] = tmp[i] ^ prev[i];
        memcpy(prev, in + offset, bs);
    }

    /* Strip PKCS#7 padding */
    uint8_t pad = out[len - 1];
    if (pad == 0 || pad > bs) return (size_t)-1;
    for (size_t i = len - pad; i < len; i++)
        if (out[i] != pad) return (size_t)-1;

    return len - pad;
}

/* ── CTR helpers ───────────────────────────────────────────────── */

/* Big-endian counter increment — wraps around on overflow */
static void increment_be(uint8_t *ctr, size_t bs) {
    for (int i = (int)bs - 1; i >= 0; i--) {
        if (++ctr[i] != 0) break;   /* no carry, done */
        /* carry propagates to next byte */
    }
}

/* ── CTR crypt (enc == dec; symmetric) ─────────────────────────── */
void ctr_crypt(block_fn_t enc, const void *ctx, size_t bs,
               const uint8_t *iv,
               const uint8_t *in, size_t len, uint8_t *out) {
    uint8_t ctr[64];     /* current counter block  */
    uint8_t ks[64];      /* keystream block        */
    memcpy(ctr, iv, bs);

    size_t offset = 0;

    /* Full blocks */
    while (offset + bs <= len) {
        enc(ctx, ctr, ks);
        for (size_t i = 0; i < bs; i++)
            out[offset + i] = in[offset + i] ^ ks[i];
        increment_be(ctr, bs);
        offset += bs;
    }

    /* Partial trailing block */
    if (offset < len) {
        enc(ctx, ctr, ks);
        for (size_t i = 0; offset + i < len; i++)
            out[offset + i] = in[offset + i] ^ ks[i];
    }
}
