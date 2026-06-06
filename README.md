# ternary-memory-pool

**Memory allocation that understands ternary data — buddy systems, trit-aligned pools, and fragmentation tracking for GPU workloads.**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Why This Exists

Memory allocation on a GPU is a battlefield. You can't call `malloc` — the GPU's memory is a scarce, high-bandwidth resource that must be managed explicitly. Memory pools, buddy allocators, and free lists are the standard tools.

Ternary neural networks add a twist: their fundamental unit isn't a byte or a float. It's a **trit** — a value in {-1, 0, +1} that occupies log₂(3) ≈ 1.58 bits. Packed 16 to a u32 word, ternary data has irregular alignment boundaries that standard allocators don't understand.

This crate provides three allocators designed for ternary GPU workloads, plus a statistics layer that tracks fragmentation, utilization, and peak usage — all in pure Rust with no GPU dependency.

## The Key Insight

Ternary weights are packed: 16 trits per u32 word. That's 16× denser than float32. A 1 GB model becomes 62.5 MB. But this density creates an alignment problem: allocations must start at word boundaries (every 16 trits), and the allocator needs to understand that a "256-element" tensor needs 16 words (256 trits = 16 × 16 trit words).

Standard allocators round to power-of-2 byte boundaries. A ternary allocator rounds to **trit-word boundaries**. This crate provides both, so you can choose the right tool for each allocation.

## Quick Start

```toml
[dependencies]
ternary-memory-pool = "0.1"
```

```rust
use ternary_memory_pool::*;

// ── Block Allocator: Fixed-Size Pool ──
let mut pool = BlockAllocator::new(256, 100); // 100 blocks × 256 bytes
let a = pool.allocate().unwrap();  // block 0
let b = pool.allocate().unwrap();  // block 1
pool.deallocate(a);                // returns to free list
let c = pool.allocate().unwrap();  // gets block 0 (LIFO reuse)

// ── Buddy Allocator: Power-of-2 Splitting ──
let mut buddy = BuddyAllocator::new(4096, 64); // 4 KB arena, min block 64 B
let a = buddy.allocate(128).unwrap();
let b = buddy.allocate(64).unwrap();
buddy.deallocate(a);  // may coalesce with buddy
let c = buddy.allocate(256).unwrap(); // uses coalesced space

// ── Ternary Allocator: Trit-Aligned ──
let mut ta = TernaryAllocator::new(1024); // 1024 trits capacity
let h1 = ta.allocate(32).unwrap();       // 32 trits (2 words)
let h2 = ta.allocate(17).unwrap();       // rounds to 32 trits (2 words)
let (start, len) = ta.get(h1).unwrap();
assert_eq!(start % 16, 0);               // word-aligned

ta.deallocate(h1);                        // coalesces adjacent free regions

// ── High-Level Pool with Statistics ──
let mut pool = MemoryPool::new(16384, 64); // 16 KB arena
pool.allocate(256);
pool.allocate(512);
let stats = pool.statistics();
println!("{}", stats);
println!("Fragmentation: {:.1}%", pool.fragmentation_ratio() * 100.0);
println!("Utilization:   {:.1}%", pool.utilization() * 100.0);
```

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
│  16 trits per u32 word                                │
└──────────────────────────────────────────────────────┘
```

## Allocator Guide

### BlockAllocator — Fixed-Size, O(1), No Frag

Pre-allocate N identical blocks. Hand them out from a free list. Return them to the free list on dealloc. O(1) for both operations. Zero fragmentation (all blocks are the same size).

**When to use:** All your allocations are the same size — fixed-size kernel tiles, uniform tensor chunks, layer activations of identical shape.

**Trade-off:** Wastes memory if allocations vary in size (a 64-byte request gets a 256-byte block).

### BuddyAllocator — Variable Size, Bounded Frag

Classic buddy system: start with one large block, split in half recursively until you reach the requested size. Freed blocks coalesce with their buddy if it's also free.

```
Arena: [========================================] 4096 bytes
Split: [==================][==================]  2 × 2048
Split: [========][========]                      2 × 1024
Alloc: [  a:512 ][  free  ][========][==========]
```

**When to use:** Variable-size allocations — different tensor dimensions, workspace buffers of different sizes. The buddy system guarantees bounded external fragmentation (worst case: 50% internal fragmentation due to power-of-2 rounding).

**Trade-off:** Internal fragmentation. Allocating 65 bytes uses 128 bytes (next power of 2). For ternary data, this can be significant.

### TernaryAllocator — Trit-Word Aligned, First-Fit

Designed for packed ternary data. All allocations are:
- Aligned to 16-trit boundaries (one u32 word)
- Sized in multiples of 16 trits
- Coalesced with adjacent free regions on deallocation

```rust
// 17 trits → rounds up to 32 trits (2 words)
// 32 trits → exactly 2 words, no rounding
// 1024 trits → 64 words
```

**When to use:** Ternary-packed weight matrices, activation buffers, gradient storage — anything measured in trits, not bytes.

**Trade-off:** First-fit can leave small free fragments. The coalescing on free mitigates this, but heavily interleaved alloc/dealloc patterns can still fragment.

### MemoryPool — Statistics Layer

Wraps a BuddyAllocator with tracking:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Fragmentation ratio | `1 - (largest_free / total_free)` | 0 = one contiguous block, 1 = fully scattered |
| Utilization | `allocated / total` | Fraction of arena in use |
| Peak bytes | max historical allocated | High-water mark for capacity planning |

## API Reference

### BlockAllocator

```rust
struct BlockAllocator { /* ... */ }
impl BlockAllocator {
    fn new(block_size: usize, count: usize) -> Self;
    fn allocate(&mut self) -> Option<usize>;
    fn deallocate(&mut self, idx: usize) -> bool;
    fn allocated_count(&self) -> usize;
    fn free_count(&self) -> usize;
    fn block_size(&self) -> usize;
    fn total_bytes(&self) -> usize;
    fn used_bytes(&self) -> usize;
}
```

### BuddyAllocator

```rust
struct BuddyAllocator { /* ... */ }
impl BuddyAllocator {
    fn new(arena_size: usize, min_block: usize) -> Self;  // both must be power of 2
    fn allocate(&mut self, size: usize) -> Option<usize>;  // returns byte offset
    fn deallocate(&mut self, offset: usize) -> bool;
    fn arena_size(&self) -> usize;
    fn allocated_bytes(&self) -> usize;
    fn allocated_count(&self) -> usize;
    fn free_bytes(&self) -> usize;
    fn free_block_counts(&self) -> Vec<(usize, usize)>;  // (order, count)
}
```

### TernaryAllocator

```rust
const TRITS_PER_WORD: usize = 16;

struct TernaryAllocator { /* ... */ }
impl TernaryAllocator {
    fn new(total_trits: usize) -> Self;         // rounds up to word boundary
    fn allocate(&mut self, trit_count: usize) -> Option<u64>;  // returns handle
    fn deallocate(&mut self, handle: u64) -> bool;
    fn get(&self, handle: u64) -> Option<(usize, usize)>;      // (start, length)
    fn total_trits(&self) -> usize;
    fn used_trits(&self) -> usize;
    fn free_trits(&self) -> usize;
    fn live_allocations(&self) -> usize;
}
```

### MemoryPool

```rust
struct MemoryPool { /* ... */ }
impl MemoryPool {
    fn new(arena_size: usize, min_block: usize) -> Self;
    fn allocate(&mut self, size: usize) -> Option<usize>;
    fn deallocate(&mut self, offset: usize) -> bool;
    fn statistics(&self) -> &PoolStatistics;
    fn fragmentation_ratio(&self) -> f64;
    fn utilization(&self) -> f64;
}

struct PoolStatistics {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub current_allocations: usize,
    pub total_allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub fragmentation_count: usize,
}
```

## Real-World Example: Managing GPU Memory for Ternary Inference

A self-driving car's vision system runs a ternary ResNet-50 on an NVIDIA T4 (16 GB VRAM). Multiple cameras share the GPU, each running inference at 30 FPS. Memory must be allocated and freed in real-time without fragmentation building up.

```rust
// Pre-allocate a memory pool for the entire model
let mut pool = MemoryPool::new(1024 * 1024 * 64, 64); // 64 MB arena

// Each camera frame needs workspace memory
let weights = pool.allocate(weight_size)?;
let activations = pool.allocate(act_size)?;
let workspace = pool.allocate(work_size)?;

// Run inference...

// Free workspace (not needed between frames)
pool.deallocate(workspace);

// Check health
if pool.fragmentation_ratio() > 0.3 {
    log::warn!("Memory fragmented: {:.1}%", pool.fragmentation_ratio() * 100.0);
    // Consider defragmentation or arena reset
}
```

The buddy allocator guarantees that even with 30 alloc/dealloc cycles per second across multiple cameras, fragmentation stays bounded.

## Performance Characteristics

| Allocator | Alloc | Dealloc | Fragmentation |
|-----------|-------|---------|---------------|
| Block | O(1) | O(1) | None |
| Buddy | O(log n) | O(log n) | Bounded (≤50% internal) |
| Ternary | O(f) | O(f + c) | Depends on pattern |

*n = number of orders, f = number of free regions, c = coalescing cost*

Memory overhead:
- **Block**: O(N) for the free list (one slot per block)
- **Buddy**: O(log₂(arena/min)) for free lists per order
- **Ternary**: O(A) for allocated handle map + O(F) for free region list

## Ecosystem Connections

Memory management underpins the entire ternary GPU stack:

- [`ternary-kernel-launch`](https://github.com/SuperInstance/ternary-kernel-launch) — the kernels that operate on memory allocated here
- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — weight matrices stored in the pool
- [`ternary-conv`](https://github.com/SuperInstance/ternary-conv) — convolution workspace buffers
- [`ternary-pool`](https://github.com/SuperInstance/ternary-pool) — pooling reduces memory pressure (smaller outputs)

## Open Questions

- **Defragmentation**: The current allocators don't support compaction (moving allocated blocks to reduce fragmentation). A real GPU memory manager needs this for long-running workloads.
- **Multi-arena**: Currently one arena per allocator. Multiple arenas (e.g., one per layer size) could reduce internal fragmentation.
- **Virtual memory**: GPU virtual memory allows overcommitting physical memory. This crate assumes a fixed physical arena.
- **Async allocation**: GPU memory allocation should be non-blocking. The current API is synchronous. An async version would integrate with the stream/event model from `ternary-kernel-launch`.

## Testing

```bash
cargo test
```

29 tests covering: block alloc/dealloc/reuse, block invalid dealloc, total/used bytes, buddy basic allocation, round-up to power of 2, min block clamping, split and exhaust, coalesce on free (full and partial), invalid dealloc, allocated/free bytes, ternary basic allocation, alignment verification, exhaustion, dealloc and reuse with coalescing, total rounds up, free trits tracking, pool basic alloc/dealloc, fragmentation measurement, utilization, peak tracking, statistics display, buddy stress (64 allocs, alternating free), and ternary stress (64 allocs, full free, re-alloc).

## License

Dual-licensed under MIT or Apache-2.0.
