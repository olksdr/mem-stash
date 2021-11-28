//! This small lib currently provides the functionality to efficiently store `u64`s into and
//! retrieve from the file-based mmap data structure.
//!
//! The `Cache` is thread-safe and can be used in the async environment. All the internal data protected by [`std::sync::Mutex`].

#![cfg_attr(all(test, feature = "nightly"), feature(test))]

#[cfg(all(test, feature = "nightly"))]
extern crate test;

mod cache;

pub use cache::Result;

pub use cache::Cache;
pub use cache::CacheReader;
pub use cache::CacheWriter;
