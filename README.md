# ternary-memory-pool

**CPU-side GPU memory pool simulation for ternary neural networks.**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

---

## Overview

`ternary-memory-pool` simulates the memory allocation strategies used by GPU runtimes when executing ternary neural network workloads. It implements pooling, buddy-system, and trit-aligned allocators — all in pure Rust with no GPU dependency.

Ternary weights ({-1, 0, +1}) are typically packed into balanced-ternary representations for efficient GPU storage. This crate models how that memory is allocated, freed, pooled, and defragmented.

## Allocator Types

### `BlockAllocator` — Fixed-Size Block Pool

The simplest pooling strategy: pre-allocate N identical blocks and hand them out from a free list. O(1) alloc/dealloc.

```rust
use ternary_memory_pool::BlockAllocator;

let mut pool = BlockAllocator::new(256, 100); // 100 blocks × 256 bytes

let a = pool.allocate().unwrap();  // block index 0
let b = pool.allocate().unwrap();  // block index 1

pool.deallocate(a);                // returns to free list for reuse
let c = pool.allocate().unwrap();  // gets block 0 again (LIFO reuse)

assert_eq!(pool.free_count(), 98);
assert_eq!(pool.used_bytes(), 512);
```

**When to use**: When all allocations are the same size (e.g., fixed-size kernel tiles, uniform tensor chunks).

### `BuddyAllocator` — Power-of-2 Splitting

A classic buddy-system allocator that splits blocks in half until the requested size is reached. Freed blocks coalesce with their buddy when possible.

```rust
use ternary_memory_pool::BuddyAllocator;

let mut buddy = BuddyAllocator::new(4096, 64); // 4 KB arena, min block 64 B

let a = buddy.allocate(128).unwrap();  // 128B block
let b = buddy.allocate(64).unwrap();   // 64B block
let c = buddy.allocate(256).unwrap();  // 256B block

buddy.deallocate(a);
buddy.deallocate(b);
// a and b may coalesce if they're buddies

let d = buddy.allocate(256).unwrap();  // can now use the coalesced space
```

**When to use**: When allocation sizes vary (e.g., different tensor dimensions, workspace buffers). The buddy system guarantees bounded external fragmentation.

**How it works**:
1. Arena starts as one large block of order `log₂(arena_size)`
2. To allocate `n` bytes, round up to next power of 2, find smallest order ≥ that
3. Split higher-order blocks down to the target order
4. On free, check if the buddy is also free — if so, coalesce up

### `TernaryAllocator` — Trit-Aligned Allocation

An allocator designed for ternary-packed data. Ternary values (trits) are packed 16 per u32 word. All allocations are aligned to word boundaries.

```rust
use ternary_memory_pool::TernaryAllocator;

let mut ta = TernaryAllocator::new(1024); // 1024 trits capacity

let h1 = ta.allocate(32).unwrap();  // 32 trits (2 words)
let h2 = ta.allocate(17).unwrap();  // rounds up to 32 trits (2 words)

let (start, len) = ta.get(h1).unwrap();
assert_eq!(start % 16, 0);  // word-aligned
assert_eq!(len, 32);

ta.deallocate(h1);
// Adjacent free regions are automatically coalesced
```

**When to use**: When managing ternary-packed weight matrices, activations, or gradient buffers where alignment to trit-word boundaries matters.

### `MemoryPool` — High-Level Pool with Statistics

Combines the buddy allocator with tracking of allocation counts, peak usage, and fragmentation metrics.

```rust
use ternary_memory_pool::MemoryPool;

let mut pool = MemoryPool::new(16384, 64); // 16 KB arena

pool.allocate(256);
pool.allocate(512);

let stats = pool.statistics();
println!("{}", stats);
// PoolStatistics { allocs: 2, deallocs: 0, current: 2, bytes: 768, peak: 768 }

println!("Fragmentation: {:.1}%", pool.fragmentation_ratio() * 100.0);
println!("Utilization:   {:.1}%", pool.utilization() * 100.0);
```

## Fragmentation Tracking

The `MemoryPool` provides real-time fragmentation metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Fragmentation ratio** | `1 - (largest_free / total_free)` | 0 = one contiguous free block, 1 = fully scattered |
| **Utilization** | `allocated / total` | Fraction of arena in use |
| **Peak bytes** | max historical allocated | High-water mark for capacity planning |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     MemoryPool                        │
│  (statistics, fragmentation, utilization tracking)    │
│  ┌──────────────────────────────────────────────────┐ │
│  │              BuddyAllocator                       │ │
│  │  (power-of-2 splitting, coalescing on free)       │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│              BlockAllocator                           │
│  (fixed-size free list, O(1) alloc/dealloc)           │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│             TernaryAllocator                          │
│  (trit-word aligned, first-fit, coalescing)           │
└──────────────────────────────────────────────────────┘
```

## Research Context

In ternary neural networks (TNNs), weights are constrained to {-1, 0, +1}, enabling:
- **~16× compression** vs float32 (packed ternary representation)
- **Eliminated multiplications** (ternary matmul is just sign/shift)
- **Reduced memory bandwidth** — the dominant bottleneck in modern GPU inference

Memory pool design significantly impacts TNN inference throughput because:
1. Ternary weights have irregular alignment (log₂3 ≈ 1.58 bits per trit)
2. Different layers have wildly different sizes (wide attention layers vs narrow FFNs)
3. Memory fragmentation directly reduces effective batch size

This crate lets you prototype and benchmark allocation strategies **without a GPU**.

## Testing

```bash
cargo test
```

29 tests covering: block alloc/dealloc/reuse, buddy splitting and coalescing, ternary alignment and coalescing, pool statistics tracking, fragmentation measurement, stress tests (high-allocation-count scenarios).

## License

Dual-licensed under MIT or Apache-2.0.
