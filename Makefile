CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c99

# macOS needs Security framework for CommonCrypto
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  BENCH_LDFLAGS = -framework Security
else
  BENCH_LDFLAGS =
endif

SRCS_CIPHER = simon64.c gift128.c modes.c

.PHONY: all test bench codesize clean

all: test_vectors bench

# ── Test suite ──────────────────────────────────────────────
test_vectors: test_vectors.c $(SRCS_CIPHER) simon64.h gift128.h modes.h
	$(CC) $(CFLAGS) -o $@ test_vectors.c $(SRCS_CIPHER)

test: test_vectors
	./test_vectors

# ── Benchmark ───────────────────────────────────────────────
bench: bench.c $(SRCS_CIPHER) simon64.h gift128.h modes.h
	$(CC) $(CFLAGS) -o $@ bench.c $(SRCS_CIPHER) $(BENCH_LDFLAGS)

# ── Code size report (.text segment) ────────────────────────
codesize: simon64.o gift128.o
	@echo ""
	@echo "=== .text segment sizes (bytes) ==="
	@echo ""
	@printf "  %-20s " "simon64.c:"
	@size -m simon64.o 2>/dev/null | grep __text | awk '{print $$NF}' || \
	 size simon64.o | tail -1 | awk '{print $$1}'
	@printf "  %-20s " "gift128.c:"
	@size -m gift128.o 2>/dev/null | grep __text | awk '{print $$NF}' || \
	 size gift128.o | tail -1 | awk '{print $$1}'
	@echo ""

simon64.o: simon64.c simon64.h
	$(CC) $(CFLAGS) -c simon64.c

gift128.o: gift128.c gift128.h
	$(CC) $(CFLAGS) -c gift128.c

# ── Clean ───────────────────────────────────────────────────
clean:
	rm -f test_vectors bench *.o
