#ifndef SIMON64_H
#define SIMON64_H

#include <stdint.h>

/*
 * SIMON 64/128 — 64-bit block, 128-bit key, 44 rounds.
 * Spec: Beaulieu et al., ePrint 2013/404, Section 3.
 */

#define SIMON64_BLOCK_BYTES  8
#define SIMON64_KEY_WORDS    4     /* 128-bit key = 4 × 32-bit words */
#define SIMON64_KEY_BYTES   16
#define SIMON64_ROUNDS      44

void simon64_key_schedule(const uint32_t key[SIMON64_KEY_WORDS],
                          uint32_t rk[SIMON64_ROUNDS]);

void simon64_encrypt(const uint32_t rk[SIMON64_ROUNDS],
                     const uint8_t  pt[SIMON64_BLOCK_BYTES],
                           uint8_t  ct[SIMON64_BLOCK_BYTES]);

void simon64_decrypt(const uint32_t rk[SIMON64_ROUNDS],
                     const uint8_t  ct[SIMON64_BLOCK_BYTES],
                           uint8_t  pt[SIMON64_BLOCK_BYTES]);

#endif /* SIMON64_H */
