#![cfg_attr(all(test, feature = "nightly"), feature(test))]

#[cfg(all(test, feature = "nightly"))]
extern crate test;

mod cache;

pub use cache::Cache;
pub use cache::CacheReader;
pub use cache::CacheWriter;
