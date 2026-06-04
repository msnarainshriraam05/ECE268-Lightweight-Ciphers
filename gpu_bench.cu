/*
 * gpu_bench.cu — Pure CUDA implementation and benchmark of
 *                SIMON 64/128 and GIFT-128/128 in CTR mode.
 *
 * Each CUDA thread encrypts one block independently (CTR mode is
 * embarrassingly parallel).  Round keys and tables live in constant memory.
 *
 * Build:  nvcc -O2 -o gpu_bench gpu_bench.cu
 * Run:    ./gpu_bench
 *
 * Requires an NVIDIA GPU with CUDA Compute Capability >= 3.5.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

/* ══════════════════════════════════════════════════════════════════
 *  SIMON 64/128 — GPU Implementation
 * ══════════════════════════════════════════════════════════════════ */

__constant__ uint32_t d_simon_rk[44];

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ uint32_t simon_f(uint32_t x) {
    return (rotl32(x, 1) & rotl32(x, 8)) ^ rotl32(x, 2);
}

__device__ void simon64_encrypt_dev(const uint32_t *rk,
                                    uint32_t *x, uint32_t *y) {
    for (int i = 0; i < 44; i++) {
        uint32_t tmp = *x;
        *x = *y ^ simon_f(*x) ^ rk[i];
        *y = tmp;
    }
}

/*
 * CTR kernel: thread idx encrypts counter block (nonce_hi, nonce_lo + idx)
 * and XORs the keystream with input data.
 */
__global__ void simon64_ctr_kernel(const uint8_t *in, uint8_t *out,
                                   uint32_t nonce_hi, uint32_t nonce_lo,
                                   uint64_t num_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    uint32_t x = nonce_hi;
    uint32_t y = nonce_lo + (uint32_t)idx;

    simon64_encrypt_dev(d_simon_rk, &x, &y);

    size_t off = idx * 8;
    uint8_t ks[8] = {
        (uint8_t)(x>>24), (uint8_t)(x>>16), (uint8_t)(x>>8), (uint8_t)x,
        (uint8_t)(y>>24), (uint8_t)(y>>16), (uint8_t)(y>>8), (uint8_t)y
    };
    for (int i = 0; i < 8; i++)
        out[off + i] = in[off + i] ^ ks[i];
}

/* ECB kernel: each thread encrypts one 8-byte block in-place */
__global__ void simon64_ecb_kernel(const uint8_t *in, uint8_t *out,
                                   uint64_t num_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    size_t off = idx * 8;
    uint32_t x = ((uint32_t)in[off]<<24) | ((uint32_t)in[off+1]<<16) |
                 ((uint32_t)in[off+2]<<8) | in[off+3];
    uint32_t y = ((uint32_t)in[off+4]<<24) | ((uint32_t)in[off+5]<<16) |
                 ((uint32_t)in[off+6]<<8) | in[off+7];

    simon64_encrypt_dev(d_simon_rk, &x, &y);

    out[off]   = x>>24; out[off+1] = x>>16; out[off+2] = x>>8; out[off+3] = x;
    out[off+4] = y>>24; out[off+5] = y>>16; out[off+6] = y>>8; out[off+7] = y;
}

/* Host key schedule */
static void simon64_key_schedule_host(const uint32_t key[4], uint32_t rk[44]) {
    static const uint64_t z3 = UINT64_C(0xfc2ce51207a635db);
    rk[0] = key[0]; rk[1] = key[1]; rk[2] = key[2]; rk[3] = key[3];
    for (int i = 4; i < 44; i++) {
        uint32_t tmp = ((rk[i-1] >> 3) | (rk[i-1] << 29)) ^ rk[i-3];
        tmp ^= (tmp >> 1) | (tmp << 31);
        rk[i] = ~rk[i-4] ^ tmp ^ (uint32_t)((z3 >> ((i-4) % 62)) & 1) ^ 3u;
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  GIFT-128/128 — GPU Implementation
 * ══════════════════════════════════════════════════════════════════ */

__constant__ uint8_t d_gift_s[16] = {
    1,10,4,12,6,15,3,9,2,13,11,7,5,0,8,14
};
__constant__ uint8_t d_gift_p[128] = {
     0,33,66,99,96, 1,34,67,64,97, 2,35,32,65,98, 3,
     4,37,70,103,100,5,38,71,68,101,6,39,36,69,102,7,
     8,41,74,107,104,9,42,75,72,105,10,43,40,73,106,11,
    12,45,78,111,108,13,46,79,76,109,14,47,44,77,110,15,
    16,49,82,115,112,17,50,83,80,113,18,51,48,81,114,19,
    20,53,86,119,116,21,54,87,84,117,22,55,52,85,118,23,
    24,57,90,123,120,25,58,91,88,121,26,59,56,89,122,27,
    28,61,94,127,124,29,62,95,92,125,30,63,60,93,126,31
};
__constant__ uint8_t d_gift_rc[40] = {
    0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,
    0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0E,
    0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,
    0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
    0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A
};
__constant__ uint32_t d_gift_rk[40][2];

__device__ void gift128_encrypt_dev(uint8_t s[32]) {
    for (int r = 0; r < 40; r++) {
        /* SubCells */
        for (int i = 0; i < 32; i++) s[i] = d_gift_s[s[i]];

        /* PermBits */
        uint8_t bits[128], pbits[128];
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                bits[4*i+j] = (s[i] >> j) & 1;
        for (int i = 0; i < 128; i++)
            pbits[d_gift_p[i]] = bits[i];

        /* AddRoundKey (operates on permuted bits) */
        for (int i = 0; i < 32; i++) {
            pbits[4*i+1] ^= (d_gift_rk[r][0] >> i) & 1;
            pbits[4*i+2] ^= (d_gift_rk[r][1] >> i) & 1;
        }
        pbits[3]   ^=  d_gift_rc[r]       & 1;
        pbits[7]   ^= (d_gift_rc[r] >> 1) & 1;
        pbits[11]  ^= (d_gift_rc[r] >> 2) & 1;
        pbits[15]  ^= (d_gift_rc[r] >> 3) & 1;
        pbits[19]  ^= (d_gift_rc[r] >> 4) & 1;
        pbits[23]  ^= (d_gift_rc[r] >> 5) & 1;
        pbits[127] ^= 1;

        /* Repack nibbles */
        for (int i = 0; i < 32; i++) {
            s[i] = 0;
            for (int j = 0; j < 4; j++)
                s[i] |= pbits[4*i+j] << j;
        }
    }
}

/*
 * CTR kernel: thread idx encrypts counter block and XORs with data.
 * Counter is placed in the lowest 4 bytes of the 16-byte nonce.
 */
__global__ void gift128_ctr_kernel(const uint8_t *in, uint8_t *out,
                                   const uint8_t *nonce,
                                   uint64_t num_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    /* Build counter block: nonce[0..11] || (big-endian counter) */
    uint8_t ctr_block[32]; /* nibble representation */
    uint8_t ctr_bytes[16];
    for (int i = 0; i < 12; i++) ctr_bytes[i] = nonce[i];
    uint32_t ctr_val = (uint32_t)idx;
    ctr_bytes[12] = ctr_val >> 24;
    ctr_bytes[13] = ctr_val >> 16;
    ctr_bytes[14] = ctr_val >> 8;
    ctr_bytes[15] = ctr_val;

    /* bytes → nibbles */
    for (int i = 0; i < 16; i++) {
        ctr_block[31 - 2*i]     = ctr_bytes[i] >> 4;
        ctr_block[31 - 2*i - 1] = ctr_bytes[i] & 0x0f;
    }

    gift128_encrypt_dev(ctr_block);

    /* nibbles → keystream bytes */
    uint8_t ks[16];
    for (int i = 0; i < 16; i++)
        ks[i] = (ctr_block[31 - 2*i] << 4) | ctr_block[30 - 2*i];

    /* XOR with input */
    size_t off = idx * 16;
    for (int i = 0; i < 16; i++)
        out[off + i] = in[off + i] ^ ks[i];
}

/* ECB kernel */
__global__ void gift128_ecb_kernel(const uint8_t *in, uint8_t *out,
                                   uint64_t num_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    size_t off = idx * 16;
    uint8_t s[32];
    for (int i = 0; i < 16; i++) {
        s[31 - 2*i]     = in[off + i] >> 4;
        s[31 - 2*i - 1] = in[off + i] & 0x0f;
    }

    gift128_encrypt_dev(s);

    for (int i = 0; i < 16; i++)
        out[off + i] = (s[31 - 2*i] << 4) | s[30 - 2*i];
}

/* Host GIFT key schedule (same as gift128.c) */
static void gift128_key_schedule_host(const uint8_t key[16],
                                      uint32_t rk[40][2]) {
    uint8_t kstate[32];
    for (int i = 0; i < 16; i++) {
        kstate[31 - 2*i]     = key[i] >> 4;
        kstate[31 - 2*i - 1] = key[i] & 0x0f;
    }
    for (int r = 0; r < 40; r++) {
        uint8_t kbits[128];
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++)
                kbits[4*i+j] = (kstate[i] >> j) & 1;

        rk[r][0] = rk[r][1] = 0;
        for (int b = 0; b < 32; b++) {
            rk[r][0] |= (uint32_t)kbits[b]      << b;
            rk[r][1] |= (uint32_t)kbits[b + 64] << b;
        }

        /* Key update */
        uint8_t tmp[32];
        for (int i = 0; i < 32; i++) tmp[i] = kstate[(i+8)%32];
        for (int i = 0; i < 24; i++) kstate[i] = tmp[i];
        kstate[24]=tmp[27]; kstate[25]=tmp[24];
        kstate[26]=tmp[25]; kstate[27]=tmp[26];
        kstate[28]=((tmp[28]&0xc)>>2)^((tmp[29]&0x3)<<2);
        kstate[29]=((tmp[29]&0xc)>>2)^((tmp[30]&0x3)<<2);
        kstate[30]=((tmp[30]&0xc)>>2)^((tmp[31]&0x3)<<2);
        kstate[31]=((tmp[31]&0xc)>>2)^((tmp[28]&0x3)<<2);
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  CPU reference implementations (for comparison)
 * ══════════════════════════════════════════════════════════════════ */

static void simon64_encrypt_cpu(const uint32_t rk[44],
                                const uint8_t pt[8], uint8_t ct[8]) {
    uint32_t x = ((uint32_t)pt[0]<<24)|((uint32_t)pt[1]<<16)|
                 ((uint32_t)pt[2]<<8)|pt[3];
    uint32_t y = ((uint32_t)pt[4]<<24)|((uint32_t)pt[5]<<16)|
                 ((uint32_t)pt[6]<<8)|pt[7];
    for (int i = 0; i < 44; i++) {
        uint32_t tmp = x;
        uint32_t f = (((x<<1)|(x>>31)) & ((x<<8)|(x>>24))) ^ ((x<<2)|(x>>30));
        x = y ^ f ^ rk[i];
        y = tmp;
    }
    ct[0]=x>>24; ct[1]=x>>16; ct[2]=x>>8; ct[3]=x;
    ct[4]=y>>24; ct[5]=y>>16; ct[6]=y>>8; ct[7]=y;
}

static void gift128_encrypt_cpu(const uint32_t rk[40][2],
                                const uint8_t pt[16], uint8_t ct[16]) {
    static const uint8_t S[16]={1,10,4,12,6,15,3,9,2,13,11,7,5,0,8,14};
    static const uint8_t P[128]={
         0,33,66,99,96, 1,34,67,64,97, 2,35,32,65,98, 3,
         4,37,70,103,100,5,38,71,68,101,6,39,36,69,102,7,
         8,41,74,107,104,9,42,75,72,105,10,43,40,73,106,11,
        12,45,78,111,108,13,46,79,76,109,14,47,44,77,110,15,
        16,49,82,115,112,17,50,83,80,113,18,51,48,81,114,19,
        20,53,86,119,116,21,54,87,84,117,22,55,52,85,118,23,
        24,57,90,123,120,25,58,91,88,121,26,59,56,89,122,27,
        28,61,94,127,124,29,62,95,92,125,30,63,60,93,126,31};
    static const uint8_t RC[40]={
        0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,
        0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0E,
        0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,
        0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
        0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A};

    uint8_t s[32];
    for (int i = 0; i < 16; i++) {
        s[31-2*i]   = pt[i] >> 4;
        s[31-2*i-1] = pt[i] & 0x0f;
    }
    for (int r = 0; r < 40; r++) {
        for (int i = 0; i < 32; i++) s[i] = S[s[i]];
        uint8_t bits[128], pb[128];
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++) bits[4*i+j]=(s[i]>>j)&1;
        for (int i = 0; i < 128; i++) pb[P[i]]=bits[i];
        for (int i = 0; i < 32; i++) {
            pb[4*i+1] ^= (rk[r][0]>>i)&1;
            pb[4*i+2] ^= (rk[r][1]>>i)&1;
        }
        pb[3]^=RC[r]&1; pb[7]^=(RC[r]>>1)&1; pb[11]^=(RC[r]>>2)&1;
        pb[15]^=(RC[r]>>3)&1; pb[19]^=(RC[r]>>4)&1; pb[23]^=(RC[r]>>5)&1;
        pb[127]^=1;
        for (int i = 0; i < 32; i++) {
            s[i]=0;
            for (int j = 0; j < 4; j++) s[i]|=pb[4*i+j]<<j;
        }
    }
    for (int i = 0; i < 16; i++)
        ct[i] = (s[31-2*i]<<4)|s[30-2*i];
}

/* ══════════════════════════════════════════════════════════════════
 *  Timing helper
 * ══════════════════════════════════════════════════════════════════ */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ══════════════════════════════════════════════════════════════════
 *  Correctness verification
 * ══════════════════════════════════════════════════════════════════ */

static int verify_simon_gpu(const uint32_t rk[44]) {
    const uint8_t pt[8] = {0x65,0x6b,0x69,0x6c,0x20,0x64,0x6e,0x75};
    const uint8_t want[8]= {0x44,0xc8,0xfc,0x20,0xb9,0xdf,0xa0,0x7a};

    uint8_t *d_in, *d_out;
    cudaMalloc(&d_in, 8);
    cudaMalloc(&d_out, 8);
    cudaMemcpy(d_in, pt, 8, cudaMemcpyHostToDevice);

    simon64_ecb_kernel<<<1, 1>>>(d_in, d_out, 1);
    cudaDeviceSynchronize();

    uint8_t got[8];
    cudaMemcpy(got, d_out, 8, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    if (memcmp(got, want, 8) != 0) {
        printf("  [FAIL] SIMON GPU ECB\n");
        return 0;
    }
    printf("  [PASS] SIMON GPU ECB matches spec vector\n");
    return 1;
}

static int verify_gift_gpu(void) {
    const uint8_t pt[16] = {0};
    const uint8_t want[16]= {
        0xcd,0x0b,0xd7,0x38,0x38,0x8a,0xd3,0xf6,
        0x68,0xb1,0x5a,0x36,0xce,0xb6,0xff,0x92
    };

    uint8_t *d_in, *d_out;
    cudaMalloc(&d_in, 16);
    cudaMalloc(&d_out, 16);
    cudaMemcpy(d_in, pt, 16, cudaMemcpyHostToDevice);

    gift128_ecb_kernel<<<1, 1>>>(d_in, d_out, 1);
    cudaDeviceSynchronize();

    uint8_t got[16];
    cudaMemcpy(got, d_out, 16, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    if (memcmp(got, want, 16) != 0) {
        printf("  [FAIL] GIFT GPU ECB\n");
        return 0;
    }
    printf("  [PASS] GIFT GPU ECB matches spec vector\n");
    return 1;
}

/* ══════════════════════════════════════════════════════════════════
 *  Main benchmark
 * ══════════════════════════════════════════════════════════════════ */

#define DATA_SIZES_COUNT 3
static const size_t DATA_SIZES[DATA_SIZES_COUNT] = {
    64 * 1024,        /* 64 KiB  */
    1024 * 1024,      /* 1 MiB   */
    16 * 1024 * 1024  /* 16 MiB  */
};
#define THREADS_PER_BLOCK 256

int main(void) {
    /* ── Key setup ──────────────────────────────────────────── */
    const uint32_t simon_key[4] = {0x03020100,0x0b0a0908,0x13121110,0x1b1a1918};
    uint32_t simon_rk[44];
    simon64_key_schedule_host(simon_key, simon_rk);
    cudaMemcpyToSymbol(d_simon_rk, simon_rk, sizeof(simon_rk));

    const uint8_t gift_key[16] = {0};
    uint32_t gift_rk[40][2];
    gift128_key_schedule_host(gift_key, gift_rk);
    cudaMemcpyToSymbol(d_gift_rk, gift_rk, sizeof(gift_rk));

    /* ── Correctness check ──────────────────────────────────── */
    printf("\n=== GPU Correctness Verification ===\n");
    int ok = 1;
    ok &= verify_simon_gpu(simon_rk);
    ok &= verify_gift_gpu();
    if (!ok) { printf("\nGPU verification failed. Aborting.\n"); return 1; }

    /* ── Benchmark ──────────────────────────────────────────── */
    printf("\n╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║              GPU vs CPU Benchmark — CTR Mode                        ║\n");
    printf("╠══════════════╦═════════╦═══════════════╦═══════════════╦═════════════╣\n");
    printf("║ Cipher       ║ Data    ║ CPU (ms)      ║ GPU (ms)      ║ Speedup     ║\n");
    printf("╠══════════════╬═════════╬═══════════════╬═══════════════╬═════════════╣\n");

    for (int ds = 0; ds < DATA_SIZES_COUNT; ds++) {
        size_t data_size = DATA_SIZES[ds];
        const char *size_label = (ds == 0) ? "64 KiB" :
                                 (ds == 1) ? "1 MiB " : "16 MiB";

        uint8_t *h_in  = (uint8_t *)malloc(data_size);
        uint8_t *h_out = (uint8_t *)malloc(data_size);
        for (size_t i = 0; i < data_size; i++) h_in[i] = (uint8_t)(i * 0x9d + 0x37);

        uint8_t *d_in, *d_out;
        cudaMalloc(&d_in, data_size);
        cudaMalloc(&d_out, data_size);
        cudaMemcpy(d_in, h_in, data_size, cudaMemcpyHostToDevice);

        /* ── SIMON 64/128 ──────────────────────────────── */
        {
            uint64_t nblocks = data_size / 8;
            int grid = (int)((nblocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

            /* CPU */
            double t0 = time_ms();
            for (uint64_t b = 0; b < nblocks; b++) {
                uint8_t ctr[8] = {0,0,0,0, (uint8_t)(b>>24),(uint8_t)(b>>16),
                                  (uint8_t)(b>>8),(uint8_t)b};
                uint8_t ks[8];
                simon64_encrypt_cpu(simon_rk, ctr, ks);
                for (int i = 0; i < 8; i++)
                    h_out[b*8+i] = h_in[b*8+i] ^ ks[i];
            }
            double cpu_ms = time_ms() - t0;

            /* GPU */
            cudaDeviceSynchronize();
            t0 = time_ms();
            simon64_ctr_kernel<<<grid, THREADS_PER_BLOCK>>>(
                d_in, d_out, 0, 0, nblocks);
            cudaDeviceSynchronize();
            double gpu_ms = time_ms() - t0;

            printf("║ SIMON 64/128 ║ %s  ║ %10.2f    ║ %10.2f    ║ %8.1fx    ║\n",
                   size_label, cpu_ms, gpu_ms,
                   gpu_ms > 0.001 ? cpu_ms / gpu_ms : 0.0);
        }

        /* ── GIFT-128/128 ──────────────────────────────── */
        {
            uint64_t nblocks = data_size / 16;
            int grid = (int)((nblocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

            uint8_t nonce[16] = {0};
            uint8_t *d_nonce;
            cudaMalloc(&d_nonce, 16);
            cudaMemcpy(d_nonce, nonce, 16, cudaMemcpyHostToDevice);

            /* CPU */
            double t0 = time_ms();
            for (uint64_t b = 0; b < nblocks; b++) {
                uint8_t ctr[16] = {0};
                ctr[12]=(uint8_t)(b>>24); ctr[13]=(uint8_t)(b>>16);
                ctr[14]=(uint8_t)(b>>8);  ctr[15]=(uint8_t)b;
                uint8_t ks[16];
                gift128_encrypt_cpu(gift_rk, ctr, ks);
                for (int i = 0; i < 16; i++)
                    h_out[b*16+i] = h_in[b*16+i] ^ ks[i];
            }
            double cpu_ms = time_ms() - t0;

            /* GPU */
            cudaDeviceSynchronize();
            t0 = time_ms();
            gift128_ctr_kernel<<<grid, THREADS_PER_BLOCK>>>(
                d_in, d_out, d_nonce, nblocks);
            cudaDeviceSynchronize();
            double gpu_ms = time_ms() - t0;

            printf("║ GIFT-128     ║ %s  ║ %10.2f    ║ %10.2f    ║ %8.1fx    ║\n",
                   size_label, cpu_ms, gpu_ms,
                   gpu_ms > 0.001 ? cpu_ms / gpu_ms : 0.0);

            cudaFree(d_nonce);
        }

        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
    }

    printf("╚══════════════╩═════════╩═══════════════╩═══════════════╩═════════════╝\n\n");

    /* ── Device info ───────────────────────────────────────── */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MHz)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000.0);
    printf("Threads per block: %d\n\n", THREADS_PER_BLOCK);

    return 0;
}
