#ifndef MODES_H
#define MODES_H

#include <stddef.h>
#include <stdint.h>

typedef void (*block_fn_t)(const void *ctx,
                           const uint8_t *in,
                           uint8_t *out);

size_t cbc_encrypt(block_fn_t enc, const void *ctx, size_t bs,
                   const uint8_t *iv,
                   const uint8_t *in, size_t len, uint8_t *out);

size_t cbc_decrypt(block_fn_t dec, const void *ctx, size_t bs,
                   const uint8_t *iv,
                   const uint8_t *in, size_t len, uint8_t *out);

void ctr_crypt(block_fn_t enc, const void *ctx, size_t bs,
               const uint8_t *iv,
               const uint8_t *in, size_t len, uint8_t *out);

#endif
