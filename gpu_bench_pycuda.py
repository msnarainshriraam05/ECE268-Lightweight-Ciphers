#!/usr/bin/env python3
"""
gpu_bench_pycuda.py — PyCUDA implementation of SIMON 64/128 and GIFT-128/128
                      CTR-mode benchmark (GPU vs CPU).

Usage:
    pip install pycuda numpy
    python gpu_bench_pycuda.py

Requires an NVIDIA GPU with CUDA drivers installed.
"""

import numpy as np
import time
import struct

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False
    print("PyCUDA not available. Install with: pip install pycuda")
    print("Also requires NVIDIA GPU and CUDA toolkit.\n")

# ══════════════════════════════════════════════════════════════
#  CUDA kernel source (embedded)
# ══════════════════════════════════════════════════════════════

CUDA_SOURCE = r"""
#include <stdint.h>

/* ── SIMON 64/128 ──────────────────────────────────────────── */

__constant__ uint32_t d_simon_rk[44];

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}
__device__ __forceinline__ uint32_t simon_f(uint32_t x) {
    return (rotl32(x, 1) & rotl32(x, 8)) ^ rotl32(x, 2);
}

__global__ void simon64_ctr_kernel(const uint8_t *in, uint8_t *out,
                                   uint32_t nonce_hi, uint32_t nonce_lo,
                                   uint64_t num_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    uint32_t x = nonce_hi, y = nonce_lo + (uint32_t)idx;
    for (int i = 0; i < 44; i++) {
        uint32_t tmp = x;
        x = y ^ simon_f(x) ^ d_simon_rk[i];
        y = tmp;
    }

    size_t off = idx * 8;
    uint8_t ks[8] = {
        (uint8_t)(x>>24),(uint8_t)(x>>16),(uint8_t)(x>>8),(uint8_t)x,
        (uint8_t)(y>>24),(uint8_t)(y>>16),(uint8_t)(y>>8),(uint8_t)y
    };
    for (int i = 0; i < 8; i++) out[off+i] = in[off+i] ^ ks[i];
}

__global__ void simon64_ecb_kernel(const uint8_t *in, uint8_t *out,
                                   uint64_t num_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    size_t off = idx * 8;
    uint32_t x = ((uint32_t)in[off]<<24)|((uint32_t)in[off+1]<<16)|
                 ((uint32_t)in[off+2]<<8)|in[off+3];
    uint32_t y = ((uint32_t)in[off+4]<<24)|((uint32_t)in[off+5]<<16)|
                 ((uint32_t)in[off+6]<<8)|in[off+7];
    for (int i = 0; i < 44; i++) {
        uint32_t tmp = x;
        x = y ^ simon_f(x) ^ d_simon_rk[i];
        y = tmp;
    }
    out[off]=x>>24; out[off+1]=x>>16; out[off+2]=x>>8; out[off+3]=x;
    out[off+4]=y>>24; out[off+5]=y>>16; out[off+6]=y>>8; out[off+7]=y;
}

/* ── GIFT-128/128 ──────────────────────────────────────────── */

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
        for (int i = 0; i < 32; i++) s[i] = d_gift_s[s[i]];
        uint8_t bits[128], pb[128];
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 4; j++) bits[4*i+j]=(s[i]>>j)&1;
        for (int i = 0; i < 128; i++) pb[d_gift_p[i]]=bits[i];
        for (int i = 0; i < 32; i++) {
            pb[4*i+1] ^= (d_gift_rk[r][0]>>i)&1;
            pb[4*i+2] ^= (d_gift_rk[r][1]>>i)&1;
        }
        pb[3]^=d_gift_rc[r]&1; pb[7]^=(d_gift_rc[r]>>1)&1;
        pb[11]^=(d_gift_rc[r]>>2)&1; pb[15]^=(d_gift_rc[r]>>3)&1;
        pb[19]^=(d_gift_rc[r]>>4)&1; pb[23]^=(d_gift_rc[r]>>5)&1;
        pb[127]^=1;
        for (int i = 0; i < 32; i++) {
            s[i]=0;
            for (int j = 0; j < 4; j++) s[i]|=pb[4*i+j]<<j;
        }
    }
}

__global__ void gift128_ctr_kernel(const uint8_t *in, uint8_t *out,
                                   const uint8_t *nonce,
                                   uint64_t num_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    uint8_t ctr_bytes[16];
    for (int i = 0; i < 12; i++) ctr_bytes[i] = nonce[i];
    uint32_t c = (uint32_t)idx;
    ctr_bytes[12]=c>>24; ctr_bytes[13]=c>>16; ctr_bytes[14]=c>>8; ctr_bytes[15]=c;

    uint8_t s[32];
    for (int i = 0; i < 16; i++) {
        s[31-2*i]   = ctr_bytes[i] >> 4;
        s[31-2*i-1] = ctr_bytes[i] & 0x0f;
    }
    gift128_encrypt_dev(s);

    size_t off = idx * 16;
    for (int i = 0; i < 16; i++) {
        uint8_t ks = (s[31-2*i]<<4)|s[30-2*i];
        out[off+i] = in[off+i] ^ ks;
    }
}
"""

# ══════════════════════════════════════════════════════════════
#  CPU reference (pure Python for comparison)
# ══════════════════════════════════════════════════════════════

def rotr32(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def rotl32(x, n):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF

def simon_f_py(x):
    return (rotl32(x,1) & rotl32(x,8)) ^ rotl32(x,2)

Z3 = 0xfc2ce51207a635db

def simon64_key_schedule(key):
    rk = list(key)
    for i in range(4, 44):
        tmp = rotr32(rk[i-1], 3) ^ rk[i-3]
        tmp ^= rotr32(tmp, 1)
        rk.append((~rk[i-4] ^ tmp ^ ((Z3 >> ((i-4)%62)) & 1) ^ 3) & 0xFFFFFFFF)
    return rk

def simon64_encrypt_py(rk, pt_bytes):
    x = struct.unpack(">I", pt_bytes[:4])[0]
    y = struct.unpack(">I", pt_bytes[4:])[0]
    for i in range(44):
        x, y = (y ^ simon_f_py(x) ^ rk[i]) & 0xFFFFFFFF, x
    return struct.pack(">II", x, y)

GIFT_S = [1,10,4,12,6,15,3,9,2,13,11,7,5,0,8,14]
GIFT_P = [
     0,33,66,99,96, 1,34,67,64,97, 2,35,32,65,98, 3,
     4,37,70,103,100,5,38,71,68,101,6,39,36,69,102,7,
     8,41,74,107,104,9,42,75,72,105,10,43,40,73,106,11,
    12,45,78,111,108,13,46,79,76,109,14,47,44,77,110,15,
    16,49,82,115,112,17,50,83,80,113,18,51,48,81,114,19,
    20,53,86,119,116,21,54,87,84,117,22,55,52,85,118,23,
    24,57,90,123,120,25,58,91,88,121,26,59,56,89,122,27,
    28,61,94,127,124,29,62,95,92,125,30,63,60,93,126,31]
GIFT_RC = [0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,
           0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0E,
           0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,
           0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
           0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A]

def gift128_key_schedule(key_bytes):
    k = [0]*32
    for i in range(16):
        k[31-2*i]   = key_bytes[i] >> 4
        k[31-2*i-1] = key_bytes[i] & 0xf
    rks = []
    for r in range(40):
        kbits = [0]*128
        for i in range(32):
            for j in range(4):
                kbits[4*i+j] = (k[i]>>j)&1
        rk0 = rk1 = 0
        for b in range(32):
            rk0 |= kbits[b] << b
            rk1 |= kbits[b+64] << b
        rks.append((rk0, rk1))
        tmp = [k[(i+8)%32] for i in range(32)]
        for i in range(24): k[i]=tmp[i]
        k[24]=tmp[27]; k[25]=tmp[24]; k[26]=tmp[25]; k[27]=tmp[26]
        k[28]=((tmp[28]&0xc)>>2)^((tmp[29]&0x3)<<2)
        k[29]=((tmp[29]&0xc)>>2)^((tmp[30]&0x3)<<2)
        k[30]=((tmp[30]&0xc)>>2)^((tmp[31]&0x3)<<2)
        k[31]=((tmp[31]&0xc)>>2)^((tmp[28]&0x3)<<2)
    return rks

# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    if not HAS_PYCUDA:
        return

    mod = SourceModule(CUDA_SOURCE)
    simon_ctr = mod.get_function("simon64_ctr_kernel")
    simon_ecb = mod.get_function("simon64_ecb_kernel")
    gift_ctr  = mod.get_function("gift128_ctr_kernel")

    # Key setup
    simon_key = np.array([0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918],
                         dtype=np.uint32)
    simon_rk = simon64_key_schedule(simon_key.tolist())
    simon_rk_arr = np.array(simon_rk, dtype=np.uint32)

    d_simon_rk_ref = mod.get_global("d_simon_rk")[0]
    cuda.memcpy_htod(d_simon_rk_ref, simon_rk_arr.tobytes())

    gift_key = bytes(16)
    gift_rk = gift128_key_schedule(gift_key)
    gift_rk_flat = np.zeros((40, 2), dtype=np.uint32)
    for r in range(40):
        gift_rk_flat[r][0] = gift_rk[r][0]
        gift_rk_flat[r][1] = gift_rk[r][1]
    d_gift_rk_ref = mod.get_global("d_gift_rk")[0]
    cuda.memcpy_htod(d_gift_rk_ref, gift_rk_flat.tobytes())

    # ── Correctness ─────────────────────────────────────────
    print("\n=== GPU Correctness (PyCUDA) ===")

    pt = np.array([0x65,0x6b,0x69,0x6c,0x20,0x64,0x6e,0x75], dtype=np.uint8)
    want = np.array([0x44,0xc8,0xfc,0x20,0xb9,0xdf,0xa0,0x7a], dtype=np.uint8)
    d_in = cuda.mem_alloc(8)
    d_out = cuda.mem_alloc(8)
    cuda.memcpy_htod(d_in, pt.tobytes())
    simon_ecb(d_in, d_out, np.uint64(1), block=(1,1,1), grid=(1,1))
    got = np.empty(8, dtype=np.uint8)
    cuda.memcpy_dtoh(got, d_out)
    print(f"  SIMON ECB: {'PASS' if np.array_equal(got, want) else 'FAIL'}")

    # ── Benchmark ───────────────────────────────────────────
    print("\n=== GPU vs CPU Benchmark — CTR Mode ===\n")
    TPB = 256

    for data_size in [64*1024, 1024*1024]:
        label = f"{data_size//1024} KiB" if data_size < 1024*1024 else f"{data_size//(1024*1024)} MiB"
        h_in = np.frombuffer(bytes((i*0x9d+0x37)&0xff for i in range(data_size)),
                             dtype=np.uint8)
        d_in = cuda.mem_alloc(data_size)
        d_out = cuda.mem_alloc(data_size)
        cuda.memcpy_htod(d_in, h_in.tobytes())

        # SIMON CTR
        nblocks = data_size // 8
        grid = ((nblocks + TPB - 1) // TPB, 1)

        t0 = time.perf_counter()
        for b in range(min(nblocks, 8192)):
            simon64_encrypt_py(simon_rk, struct.pack(">II", 0, b))
        cpu_per_block = (time.perf_counter() - t0) / min(nblocks, 8192)
        cpu_est = cpu_per_block * nblocks * 1000

        start = cuda.Event()
        end = cuda.Event()
        start.record()
        simon_ctr(d_in, d_out, np.uint32(0), np.uint32(0),
                  np.uint64(nblocks), block=(TPB,1,1), grid=(grid[0],1))
        end.record()
        end.synchronize()
        gpu_ms = start.time_till(end)

        speedup = cpu_est / gpu_ms if gpu_ms > 0.001 else 0
        print(f"  SIMON 64/128  {label:>7s}  CPU ~{cpu_est:8.1f} ms  GPU {gpu_ms:8.2f} ms  {speedup:6.1f}x")

        # GIFT CTR
        nblocks_g = data_size // 16
        grid_g = ((nblocks_g + TPB - 1) // TPB, 1)
        nonce = np.zeros(16, dtype=np.uint8)
        d_nonce = cuda.mem_alloc(16)
        cuda.memcpy_htod(d_nonce, nonce.tobytes())

        start.record()
        gift_ctr(d_in, d_out, d_nonce, np.uint64(nblocks_g),
                 block=(TPB,1,1), grid=(grid_g[0],1))
        end.record()
        end.synchronize()
        gpu_ms_g = start.time_till(end)

        print(f"  GIFT-128      {label:>7s}  CPU   (slow)     GPU {gpu_ms_g:8.2f} ms")

        d_in.free()
        d_out.free()

    print()
    print(f"  GPU: {cuda.Device(0).name()}")
    print(f"  Threads/block: {TPB}\n")

if __name__ == "__main__":
    main()
