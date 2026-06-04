#ifndef GIFT128_H
#define GIFT128_H

#include <stdint.h>

#define GIFT128_ROUNDS      40
#define GIFT128_BLOCK_BYTES 16
#define GIFT128_KEY_BYTES   16

void gift128_key_schedule(const uint8_t key[16], uint32_t rk[GIFT128_ROUNDS][2]);

void gift128_encrypt(const uint32_t rk[GIFT128_ROUNDS][2],
                     const uint8_t  pt[16], uint8_t ct[16]);

void gift128_decrypt(const uint32_t rk[GIFT128_ROUNDS][2],
                     const uint8_t  ct[16], uint8_t pt[16]);

#endif /* GIFT128_H */
