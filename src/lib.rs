//! # ternary-memory-pool
//!
//! CPU-side simulation of GPU memory pool management for ternary neural networks.
//! Implements pooling allocators (block, buddy, ternary-aligned) with fragmentation
//! tracking — no actual GPU memory is touched, but the algorithms are production-grade.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// BlockAllocator — fixed-size block pool
// ---------------------------------------------------------------------------

/// A simple free-list allocator that hands out fixed-size blocks from a contiguous arena.
///
/// Each block has the same capacity.  Freed blocks are pushed onto a free list
/// for O(1) reuse.
#[derive(Debug)]
pub struct BlockAllocator {
    block_size: usize,
    total_blocks: usize,
    free_list: VecDeque<usize>, // block index
    allocated: HashMap<usize, bool>,
}

impl BlockAllocator {
    /// Create a new allocator with `count` blocks of `block_size` bytes each.
    pub fn new(block_size: usize, count: usize) -> Self {
        let free_list = (0..count).collect();
        Self {
            block_size,
            total_blocks: count,
            free_list,
            allocated: HashMap::new(),
        }
    }

    /// Allocate one block. Returns its index, or `None` if exhausted.
    pub fn allocate(&mut self) -> Option<usize> {
        let idx = self.free_list.pop_front()?;
        self.allocated.insert(idx, true);
        Some(idx)
    }

    /// Deallocate a block, returning it to the free list.
    ///
    /// Returns `true` if the block was actually allocated.
    pub fn deallocate(&mut self, idx: usize) -> bool {
        if self.allocated.remove(&idx).is_some() {
            self.free_list.push_back(idx);
            true
        } else {
            false
        }
    }

    /// Number of currently allocated blocks.
    pub fn allocated_count(&self) -> usize {
        self.allocated.len()
    }

    /// Number of free blocks.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Block size in bytes.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total arena size in bytes.
    pub fn total_bytes(&self) -> usize {
        self.block_size * self.total_blocks
    }

    /// Used bytes.
    pub fn used_bytes(&self) -> usize {
        self.block_size * self.allocated_count()
    }
}

// ---------------------------------------------------------------------------
// BuddyAllocator — power-of-2 splitting
// ---------------------------------------------------------------------------

/// A buddy-system allocator over a contiguous arena of `2^max_order` bytes.
///
/// Supports allocation of power-of-2 sized blocks.  On free, buddies are
/// coalesced automatically.
#[derive(Debug)]
pub struct BuddyAllocator {
    /// Total size of the arena (bytes). Must be a power of 2.
    arena_size: usize,
    /// Minimum allocation unit (bytes). Must be a power of 2.
    min_block: usize,
    /// max_order = log2(arena_size), min_order = log2(min_block).
    max_order: usize,
    min_order: usize,
    /// Free lists: free[i] holds offsets of free blocks of order i.
    free: Vec<Vec<usize>>,
    /// Allocated blocks: offset → order.
    allocated: BTreeMap<usize, usize>,
}

impl BuddyAllocator {
    /// Create a buddy allocator.
    ///
    /// `arena_size` and `min_block` must be powers of two, and
    /// `arena_size >= min_block`.
    pub fn new(arena_size: usize, min_block: usize) -> Self {
        assert!(arena_size.is_power_of_two());
        assert!(min_block.is_power_of_two());
        assert!(arena_size >= min_block);

        let max_order = arena_size.trailing_zeros() as usize;
        let min_order = min_block.trailing_zeros() as usize;
        let num_orders = max_order + 1;

        let mut free = vec![Vec::new(); num_orders];
        free[max_order].push(0); // whole arena is free

        Self {
            arena_size,
            min_block,
            max_order,
            min_order,
            free,
            allocated: BTreeMap::new(),
        }
    }

    /// Allocate `size` bytes (rounded up to the next power of 2, clamped to min_block).
    ///
    /// Returns the byte offset of the allocated block, or `None` if insufficient space.
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        let size = size.max(self.min_block);
        let size = size.next_power_of_two();
        let order = size.trailing_zeros() as usize;
        if order < self.min_order || order > self.max_order {
            return None;
        }

        // Find a free block at order or higher
        let mut found_order = None;
        for o in order..=self.max_order {
            if !self.free[o].is_empty() {
                found_order = Some(o);
                break;
            }
        }
        let found_order = found_order?;

        // Pop the block and split down to target order
        let mut offset = self.free[found_order].pop().unwrap();
        for o in (order..found_order).rev() {
            // Split: push the upper buddy at order o
            let buddy = offset + (1usize << o);
            self.free[o].push(buddy);
        }

        self.allocated.insert(offset, order);
        Some(offset)
    }

    /// Deallocate a previously allocated block at `offset`.
    ///
    /// Returns `true` if the block was found and freed.
    pub fn deallocate(&mut self, offset: usize) -> bool {
        let order = match self.allocated.remove(&offset) {
            Some(o) => o,
            None => return false,
        };

        // Coalesce with buddy
        let mut current_offset = offset;
        let mut current_order = order;
        while current_order < self.max_order {
            let buddy = current_offset ^ (1usize << current_order);
            if let Some(pos) = self.free[current_order].iter().position(|&x| x == buddy) {
                self.free[current_order].swap_remove(pos);
                current_offset = current_offset.min(buddy);
                current_order += 1;
            } else {
                break;
            }
        }
        self.free[current_order].push(current_offset);
        true
    }

    /// Arena size in bytes.
    pub fn arena_size(&self) -> usize {
        self.arena_size
    }

    /// Total allocated bytes.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated.values().map(|&o| 1usize << o).sum()
    }

    /// Number of allocated blocks.
    pub fn allocated_count(&self) -> usize {
        self.allocated.len()
    }

    /// Total free bytes.
    pub fn free_bytes(&self) -> usize {
        self.arena_size - self.allocated_bytes()
    }

    /// Count of free blocks at each order.
    pub fn free_block_counts(&self) -> Vec<(usize, usize)> {
        self.free
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(order, v)| (order, v.len()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TernaryAllocator — trit-aligned allocation
// ---------------------------------------------------------------------------

/// An allocator that hands out trit-aligned chunks.
///
/// Ternary values (trits) are packed into balanced-ternary words.  This allocator
/// ensures that every allocation starts at a trit-word boundary (multiples of
/// `TRITS_PER_WORD` = 16) and is sized in whole words.
pub const TRITS_PER_WORD: usize = 16;

#[derive(Debug)]
pub struct TernaryAllocator {
    /// Total capacity in trits.
    total_trits: usize,
    /// Free list of (start_trit, length_trits).
    free_regions: Vec<(usize, usize)>,
    /// Allocated: handle → (start_trit, length_trits).
    allocated: HashMap<u64, (usize, usize)>,
    next_handle: u64,
}

impl TernaryAllocator {
    /// Create a ternary allocator with the given total trit capacity.
    ///
    /// Capacity is rounded up to a whole number of words.
    pub fn new(total_trits: usize) -> Self {
        let total_trits = Self::align_up(total_trits);
        Self {
            total_trits,
            free_regions: vec![(0, total_trits)],
            allocated: HashMap::new(),
            next_handle: 0,
        }
    }

    /// Round up to a trit-word boundary.
    fn align_up(trits: usize) -> usize {
        (trits + TRITS_PER_WORD - 1) / TRITS_PER_WORD * TRITS_PER_WORD
    }

    /// Allocate `trit_count` trits. Returns a handle.
    ///
    /// The size is rounded up to a word boundary.
    pub fn allocate(&mut self, trit_count: usize) -> Option<u64> {
        let needed = Self::align_up(trit_count.max(1));
        // First-fit search
        let idx = self
            .free_regions
            .iter()
            .position(|&(_, len)| len >= needed)?;

        let (start, len) = self.free_regions[idx];
        let handle = self.next_handle;
        self.next_handle += 1;
        self.allocated.insert(handle, (start, needed));

        if len == needed {
            self.free_regions.swap_remove(idx);
        } else {
            self.free_regions[idx] = (start + needed, len - needed);
        }
        Some(handle)
    }

    /// Deallocate a handle.
    pub fn deallocate(&mut self, handle: u64) -> bool {
        let (start, len) = match self.allocated.remove(&handle) {
            Some(v) => v,
            None => return false,
        };
        self.free_regions.push((start, len));
        self.coalesce();
        true
    }

    /// Merge adjacent free regions.
    fn coalesce(&mut self) {
        if self.free_regions.is_empty() {
            return;
        }
        self.free_regions.sort_by_key(|&(s, _)| s);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free_regions.len());
        let mut cur = self.free_regions[0];
        for &(s, l) in &self.free_regions[1..] {
            if cur.0 + cur.1 == s {
                cur.1 += l;
            } else {
                merged.push(cur);
                cur = (s, l);
            }
        }
        merged.push(cur);
        self.free_regions = merged;
    }

    /// Get allocation info.
    pub fn get(&self, handle: u64) -> Option<(usize, usize)> {
        self.allocated.get(&handle).copied()
    }

    /// Total trit capacity.
    pub fn total_trits(&self) -> usize {
        self.total_trits
    }

    /// Used trits.
    pub fn used_trits(&self) -> usize {
        self.allocated.values().map(|&(_, l)| l).sum()
    }

    /// Free trits.
    pub fn free_trits(&self) -> usize {
        self.total_trits - self.used_trits()
    }

    /// Number of live allocations.
    pub fn live_allocations(&self) -> usize {
        self.allocated.len()
    }
}

// ---------------------------------------------------------------------------
// MemoryPool — top-level pool with statistics
// ---------------------------------------------------------------------------

/// A memory pool that combines a buddy allocator with ternary-aware sub-allocation.
///
/// Provides high-level alloc/dealloc and statistics.
#[derive(Debug)]
pub struct MemoryPool {
    buddy: BuddyAllocator,
    stats: PoolStatistics,
}

/// Pool statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub current_allocations: usize,
    pub total_allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub fragmentation_count: usize, // number of non-contiguous free regions
}

impl MemoryPool {
    /// Create a pool with the given arena size (power of 2) and minimum block.
    pub fn new(arena_size: usize, min_block: usize) -> Self {
        Self {
            buddy: BuddyAllocator::new(arena_size, min_block),
            stats: PoolStatistics::default(),
        }
    }

    /// Allocate `size` bytes.
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        let offset = self.buddy.allocate(size)?;
        let block_size = size.max(self.buddy.min_block).next_power_of_two();
        self.stats.total_allocations += 1;
        self.stats.current_allocations += 1;
        self.stats.total_allocated_bytes += block_size;
        self.stats.peak_allocated_bytes = self.stats.peak_allocated_bytes.max(self.stats.total_allocated_bytes);
        Some(offset)
    }

    /// Deallocate a block at `offset`.
    pub fn deallocate(&mut self, offset: usize) -> bool {
        // We need to know the order to compute freed bytes — peek from allocated
        let freed = self.buddy.deallocate(offset);
        if freed {
            self.stats.total_deallocations += 1;
            self.stats.current_allocations = self.stats.current_allocations.saturating_sub(1);
        }
        freed
    }

    /// Get a snapshot of pool statistics.
    pub fn statistics(&self) -> &PoolStatistics {
        &self.stats
    }

    /// Compute fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented).
    ///
    /// Defined as `1 - (largest_free_block / total_free)`.
    pub fn fragmentation_ratio(&self) -> f64 {
        let free_counts = self.buddy.free_block_counts();
        if free_counts.is_empty() {
            return 0.0;
        }
        let total_free: usize = free_counts
            .iter()
            .map(|&(order, count)| (1usize << order) * count)
            .sum();
        if total_free == 0 {
            return 0.0;
        }
        let largest: usize = free_counts
            .iter()
            .map(|&(order, _)| 1usize << order)
            .max()
            .unwrap_or(0);
        1.0 - (largest as f64 / total_free as f64)
    }

    /// Utilization ratio: allocated / total.
    pub fn utilization(&self) -> f64 {
        if self.buddy.arena_size == 0 {
            return 0.0;
        }
        self.buddy.allocated_bytes() as f64 / self.buddy.arena_size() as f64
    }

    /// Underlying buddy allocator reference.
    pub fn buddy(&self) -> &BuddyAllocator {
        &self.buddy
    }
}

impl fmt::Display for PoolStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PoolStatistics {{ allocs: {}, deallocs: {}, current: {}, bytes: {}, peak: {} }}",
            self.total_allocations,
            self.total_deallocations,
            self.current_allocations,
            self.total_allocated_bytes,
            self.peak_allocated_bytes
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BlockAllocator ----

    #[test]
    fn block_alloc_basic() {
        let mut ba = BlockAllocator::new(256, 10);
        let a = ba.allocate().unwrap();
        let b = ba.allocate().unwrap();
        assert_ne!(a, b);
        assert_eq!(ba.allocated_count(), 2);
        assert_eq!(ba.free_count(), 8);
    }

    #[test]
    fn block_dealloc_reuse() {
        let mut ba = BlockAllocator::new(64, 3);
        let a = ba.allocate().unwrap();
        let b = ba.allocate().unwrap();
        let c = ba.allocate().unwrap();
        assert!(ba.allocate().is_none()); // full

        assert!(ba.deallocate(b));
        let d = ba.allocate().unwrap(); // should reuse b's slot
        assert_eq!(d, b);
        assert_eq!(ba.allocated_count(), 3);
    }

    #[test]
    fn block_dealloc_invalid() {
        let mut ba = BlockAllocator::new(64, 2);
        assert!(!ba.deallocate(99)); // never allocated
    }

    #[test]
    fn block_total_bytes() {
        let ba = BlockAllocator::new(128, 5);
        assert_eq!(ba.total_bytes(), 640);
    }

    #[test]
    fn block_used_bytes() {
        let mut ba = BlockAllocator::new(128, 5);
        ba.allocate();
        ba.allocate();
        assert_eq!(ba.used_bytes(), 256);
    }

    // ---- BuddyAllocator ----

    #[test]
    fn buddy_alloc_basic() {
        let mut ba = BuddyAllocator::new(1024, 64);
        let a = ba.allocate(128).unwrap();
        assert_eq!(a % 128, 0); // aligned to block size
        assert_eq!(ba.allocated_count(), 1);
    }

    #[test]
    fn buddy_alloc_round_up() {
        let mut ba = BuddyAllocator::new(1024, 64);
        // 100 rounds up to 128
        let a = ba.allocate(100).unwrap();
        assert_eq!(a % 128, 0);
    }

    #[test]
    fn buddy_alloc_min_block() {
        let mut ba = BuddyAllocator::new(1024, 64);
        // 1 byte rounds up to min_block = 64
        let a = ba.allocate(1).unwrap();
        assert_eq!(a % 64, 0);
    }

    #[test]
    fn buddy_split_and_alloc() {
        let mut ba = BuddyAllocator::new(1024, 64);
        // Allocate all 1024 as 16 × 64-byte blocks
        let mut offsets = Vec::new();
        for _ in 0..16 {
            offsets.push(ba.allocate(64).unwrap());
        }
        assert_eq!(ba.allocated_count(), 16);
        assert!(ba.allocate(64).is_none()); // exhausted
    }

    #[test]
    fn buddy_coalesce_on_free() {
        let mut ba = BuddyAllocator::new(256, 64);
        let a = ba.allocate(64).unwrap(); // order 6 (64)
        let b = ba.allocate(64).unwrap();
        assert_eq!(ba.allocated_count(), 2);
        ba.deallocate(a);
        ba.deallocate(b);
        assert_eq!(ba.allocated_count(), 0);
        // Should be able to allocate the full 256 again
        let full = ba.allocate(256).unwrap();
        assert_eq!(full, 0);
    }

    #[test]
    fn buddy_partial_coalesce() {
        let mut ba = BuddyAllocator::new(256, 64);
        let a = ba.allocate(64).unwrap(); // offset 0
        let b = ba.allocate(64).unwrap(); // offset 64
        let c = ba.allocate(64).unwrap(); // offset 128

        // Free a and b → coalesce to 128
        ba.deallocate(a);
        ba.deallocate(b);
        // Now allocate 128 — should get offset 0
        let d = ba.allocate(128).unwrap();
        assert_eq!(d, 0);
        // c is still allocated at 128, so remaining free is just 192..256 (64 bytes)
    }

    #[test]
    fn buddy_deallocate_invalid() {
        let mut ba = BuddyAllocator::new(256, 64);
        assert!(!ba.deallocate(999));
    }

    #[test]
    fn buddy_allocated_bytes() {
        let mut ba = BuddyAllocator::new(1024, 64);
        ba.allocate(64);
        ba.allocate(128);
        // 64 + 128 = 192
        assert_eq!(ba.allocated_bytes(), 192);
    }

    #[test]
    fn buddy_free_bytes() {
        let mut ba = BuddyAllocator::new(1024, 64);
        ba.allocate(128);
        assert_eq!(ba.free_bytes(), 896);
    }

    // ---- TernaryAllocator ----

    #[test]
    fn ternary_alloc_basic() {
        let mut ta = TernaryAllocator::new(160);
        let h = ta.allocate(32).unwrap();
        let (start, len) = ta.get(h).unwrap();
        assert_eq!(start % TRITS_PER_WORD, 0);
        assert_eq!(len, 32);
    }

    #[test]
    fn ternary_alloc_aligns_up() {
        let mut ta = TernaryAllocator::new(160);
        let h = ta.allocate(17).unwrap(); // rounds up to 32
        let (_, len) = ta.get(h).unwrap();
        assert_eq!(len, 32);
    }

    #[test]
    fn ternary_alloc_exhaustion() {
        let mut ta = TernaryAllocator::new(32);
        let _h1 = ta.allocate(32).unwrap();
        assert!(ta.allocate(1).is_none());
    }

    #[test]
    fn ternary_dealloc_and_reuse() {
        let mut ta = TernaryAllocator::new(64);
        let h1 = ta.allocate(32).unwrap();
        let h2 = ta.allocate(32).unwrap();
        assert!(ta.deallocate(h1));
        let h3 = ta.allocate(16).unwrap();
        assert!(ta.get(h3).unwrap().0 < 32); // reused the first region
        assert_eq!(ta.used_trits(), 48); // 32 + 16
    }

    #[test]
    fn ternary_dealloc_invalid() {
        let mut ta = TernaryAllocator::new(64);
        assert!(!ta.deallocate(999));
    }

    #[test]
    fn ternary_coalesce_adjacent() {
        let mut ta = TernaryAllocator::new(64);
        let h1 = ta.allocate(16).unwrap();
        let h2 = ta.allocate(16).unwrap();
        let h3 = ta.allocate(16).unwrap();

        ta.deallocate(h1);
        ta.deallocate(h2);
        // The two 16-trit regions should coalesce into one 32-trit region
        let h4 = ta.allocate(32).unwrap(); // needs the coalesced region
        assert_eq!(ta.used_trits(), 48); // h3(16) + h4(32)
    }

    #[test]
    fn ternary_total_rounds_up() {
        let ta = TernaryAllocator::new(20);
        assert_eq!(ta.total_trits(), 32); // rounded up to word boundary
    }

    #[test]
    fn ternary_free_trits() {
        let mut ta = TernaryAllocator::new(64);
        ta.allocate(16);
        assert_eq!(ta.free_trits(), 48);
    }

    // ---- MemoryPool ----

    #[test]
    fn pool_basic_alloc_dealloc() {
        let mut pool = MemoryPool::new(1024, 64);
        let a = pool.allocate(128).unwrap();
        assert_eq!(pool.statistics().total_allocations, 1);
        assert_eq!(pool.statistics().current_allocations, 1);
        assert!(pool.deallocate(a));
        assert_eq!(pool.statistics().total_deallocations, 1);
        assert_eq!(pool.statistics().current_allocations, 0);
    }

    #[test]
    fn pool_fragmentation_ratio() {
        let mut pool = MemoryPool::new(1024, 64);
        // Allocate then free alternating blocks to create fragmentation
        let a = pool.allocate(64).unwrap();
        let b = pool.allocate(64).unwrap();
        let c = pool.allocate(64).unwrap();
        let d = pool.allocate(64).unwrap();

        pool.deallocate(a);
        pool.deallocate(c);
        // Free blocks at offsets 0 and 128, not contiguous
        let frag = pool.fragmentation_ratio();
        assert!(frag > 0.0, "fragmentation should be > 0");
    }

    #[test]
    fn pool_utilization() {
        let mut pool = MemoryPool::new(1024, 64);
        pool.allocate(256);
        let util = pool.utilization();
        assert!(util > 0.0 && util <= 1.0);
    }

    #[test]
    fn pool_peak_tracking() {
        let mut pool = MemoryPool::new(1024, 64);
        pool.allocate(128);
        pool.allocate(256);
        assert_eq!(pool.statistics().peak_allocated_bytes, 384);
        pool.deallocate(0); // doesn't change peak
        assert_eq!(pool.statistics().peak_allocated_bytes, 384);
    }

    #[test]
    fn pool_stats_display() {
        let stats = PoolStatistics {
            total_allocations: 10,
            total_deallocations: 5,
            current_allocations: 5,
            total_allocated_bytes: 512,
            peak_allocated_bytes: 640,
            fragmentation_count: 0,
        };
        let s = format!("{}", stats);
        assert!(s.contains("allocs: 10"));
        assert!(s.contains("peak: 640"));
    }

    // ---- Stress test ----

    #[test]
    fn buddy_stress() {
        let mut ba = BuddyAllocator::new(4096, 64);
        let mut allocs: Vec<usize> = Vec::new();
        // Allocate many small blocks
        for _ in 0..64 {
            if let Some(off) = ba.allocate(64) {
                allocs.push(off);
            }
        }
        assert_eq!(allocs.len(), 64);
        assert!(ba.allocate(64).is_none());

        // Free every other one
        let mut to_free: Vec<usize> = allocs.iter().step_by(2).copied().collect();
        for off in &to_free {
            ba.deallocate(*off);
        }
        // Can allocate again
        let new = ba.allocate(64).unwrap();
        assert!(allocs.contains(&new) || to_free.contains(&new));
    }

    #[test]
    fn ternary_stress() {
        let mut ta = TernaryAllocator::new(1024);
        let mut handles: Vec<u64> = Vec::new();
        for _ in 0..64 {
            if let Some(h) = ta.allocate(16) {
                handles.push(h);
            }
        }
        assert_eq!(handles.len(), 64); // 1024 / 16 = 64
        assert!(ta.allocate(16).is_none());

        // Free all and verify we can re-allocate
        for h in handles {
            ta.deallocate(h);
        }
        assert_eq!(ta.used_trits(), 0);
        let _h = ta.allocate(1024).unwrap();
    }
}
