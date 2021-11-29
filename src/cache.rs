use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use memmap2;

/// Wrapper over the [`std::result::Result`]
pub type Result<T, E = Box<dyn std::error::Error + Sync + Send>> = std::result::Result<T, E>;

// Just some initial size we use for our file-based mmap
const PAGE_SIZE: usize = 65536;

/// Cache provides the useful functionality to efficiently store `u64`s into the file-based mmap allocation
/// and retrieve those from. It's thread safe and can be used for async access.
/// Data internal data is protected by [`std::sync::Mutex`]
#[derive(Debug, Clone)]
pub struct Cache {
    inner: Arc<Mutex<CacheInner>>,
}

impl Cache {
    /// Returns the [`Cache`] instance where all the data will be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use mem_stash::Cache;
    /// let cache = Cache::new("cache-file.dat".into()).unwrap();
    /// ```
    pub fn new(path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        file.set_len((PAGE_SIZE * std::mem::size_of::<u64>()) as u64)?;

        // Safety:
        // we make sure the file is opened and it has initial size greater than 0
        let memmap = unsafe { memmap2::MmapMut::map_mut(&file) }?;
        let inner = Arc::new(Mutex::new(CacheInner {
            file,
            memmap,
            elements_count: PAGE_SIZE,
        }));
        Ok(Self { inner })
    }

    /// Returns the index of the beginning of the data in the mmap alocatated space
    fn get_location(index: usize) -> usize {
        (index << 3) as usize
    }

    /// Accepts the function which is executed in the context of the reader.
    /// Caller must keep in mind that lock will be held on the data for the time of function
    /// execution.
    ///
    ///
    /// Try to keep the run time as short as possible
    ///
    /// # Examples
    ///
    /// ```
    /// use mem_stash::Cache;
    /// let cache = Cache::new("cache-file.dat".into()).unwrap();
    /// let data = vec![1u64, 2, 3, 4];
    ///
    /// # cache
    /// #     .with_writer(|w| {
    /// #         for (i, e) in data.iter().enumerate() {
    /// #             w.write(i, *e).unwrap();
    /// #         }
    /// #         Ok(())
    /// #     })
    /// #     .unwrap();
    ///
    /// cache
    ///     .with_reader(|r| {
    ///         for (i, e) in data.iter().enumerate() {
    ///             let result = r.read(i).unwrap(); // must check the result of the writing into the cache
    ///             assert_eq!(result, *e)
    ///         }
    ///         Ok(())
    ///     })
    ///     .unwrap(); // must consume the result of the operation
    /// ```
    pub fn with_reader<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&dyn CacheReader) -> Result<()>,
    {
        let guard = self.inner.lock();
        let reader = guard.as_deref().map_err(|e| {
            format!(
                "Cannot get the shared reference to the mmap allocation: {:?}",
                e
            )
        })?;
        f(reader)
    }

    /// Accepts the function which is executed into the context of the writer.
    ///
    /// # Examples
    ///
    /// ```
    /// use mem_stash::Cache;
    /// let cache = Cache::new("cache-file.dat".into()).unwrap();
    /// let data = vec![1u64, 2, 3, 4];
    ///
    /// cache
    ///     .with_writer(|w| {
    ///         for (i, e) in data.iter().enumerate() {
    ///             w.write(i, *e).unwrap(); // must check the result of the writing into the cache
    ///         }
    ///         Ok(())
    ///     })
    ///     .unwrap(); // must consume the result of the operation
    /// ```
    pub fn with_writer<F>(&self, f: F) -> Result<()>
    where
        F: Fn(&mut dyn CacheWriter) -> Result<()>,
    {
        let mut guard = self.inner.lock();
        let writer = guard.as_deref_mut().map_err(|e| {
            format!(
                "Cannot get a mutable reference to the mmap allocation: {:?}",
                e
            )
        })?;
        f(writer)
    }

    /// Writes the single element into the cache on the provided index
    ///
    /// # Examples
    ///
    /// ```
    /// # use mem_stash::Cache;
    /// # let cache = Cache::new("cache-file.dat".into()).unwrap();
    ///
    /// cache.write(1, 234u64).unwrap();
    ///
    /// ```
    pub fn write(&self, index: usize, value: u64) -> Result<()> {
        self.with_writer(|writer| writer.write(index, value))
    }

    /// Returns the element read from the requested index.
    ///
    /// If the provided `index` is out of bound of the allocated file error will be returned
    ///
    /// # Examples
    ///
    /// ```
    /// # use mem_stash::Cache;
    /// # let cache = Cache::new("cache-file.dat".into()).unwrap();
    ///
    /// cache.write(1, 42).unwrap();
    /// let result = cache.read(1).unwrap();
    ///
    /// assert_eq!(result, 42);
    ///
    /// ```
    pub fn read(&self, index: usize) -> Result<u64> {
        let mut result: Option<u64> = None;
        self.with_reader(|reader| {
            result = reader.read(index).ok();
            Ok(())
        })?;
        result.ok_or_else(|| format!("Unable to find requested index = {}", index).into())
    }

    /// Flushes the data to the file which backes the mmap allocation
    pub fn flush(&self) -> Result<()> {
        self.with_writer(|writer| writer.flush())
    }
}

#[derive(Debug)]
struct CacheInner {
    file: File,
    elements_count: usize,
    memmap: memmap2::MmapMut,
}

/// Provides the implementation of the reader context
pub trait CacheReader {
    /// Returns the element from the requested index
    fn read(&self, index: usize) -> Result<u64>;
}

impl CacheReader for CacheInner {
    fn read(&self, index: usize) -> Result<u64> {
        if index > self.elements_count {
            return Err(format!(
                "Unable to access index = {} out of bounds, max available index = {}",
                index, self.elements_count
            )
            .into());
        }

        let mut buf = [0u8; std::mem::size_of::<u64>()];
        // Safety:
        // We are checking if the accessed index is still in the bound of the allocated file
        // and return the error before hitting the `unsafe` code
        unsafe {
            self.memmap
                .as_ptr()
                .add(Cache::get_location(index))
                .copy_to(buf.as_mut_ptr(), std::mem::size_of::<u64>())
        };
        Ok(u64::from_le_bytes(buf))
    }
}

/// Provides the implementation of the writer context
pub trait CacheWriter {
    /// Writes the data to the mmap on the requested index.
    /// If the index bigger than current allocation, the mmap will be grown together with file on
    /// which it relies
    fn write(&mut self, index: usize, value: u64) -> Result<()>;

    /// Flushes the data to the file
    fn flush(&mut self) -> Result<()>;
}

impl CacheWriter for CacheInner {
    fn flush(&mut self) -> Result<()> {
        self.memmap.flush()?;
        Ok(())
    }

    fn write(&mut self, index: usize, value: u64) -> Result<()> {
        if self.elements_count < index {
            self.memmap.flush()?;
            let file_size = self.file.metadata()?.len();
            let new_size = (index - self.elements_count + PAGE_SIZE) * std::mem::size_of::<u64>()
                + file_size as usize;
            self.file.set_len(new_size as u64)?;
            self.elements_count += index - self.elements_count + PAGE_SIZE;

            // Safety:
            // we just resized the file and are ok to create a new MMap from the file with new size
            let new_mmap = unsafe { memmap2::MmapOptions::new().map_mut(&self.file)? };
            self.memmap = new_mmap;
        }

        let ptr = self.memmap.as_mut_ptr();
        // Safety:
        // file is big enough to write in the requested location
        unsafe {
            ptr.add(Cache::get_location(index))
                .copy_from(value.to_le_bytes().as_mut_ptr(), std::mem::size_of::<u64>());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::*;
    use rand::{prelude::SliceRandom, Rng};
    use rayon::prelude::*;
    use tempfile::tempdir;

    #[test]
    // Single thread , but we still getting lock on each write and read operation
    fn single_threaded() {
        // data to write into the cache
        let data = get_random_items(100000);

        // set up our cache file
        let dir = tempdir().unwrap();
        println!("[single] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("single-thread.dat")).expect("Mmap allocation");

        for v in data.iter() {
            cache.write(v.0, v.1).expect("Data is written to MMap");
        }

        for v in data {
            let data_from_mmap: u64 = cache.read(v.0).expect("Data is read from MMap");
            assert_eq!(data_from_mmap, v.1);
        }
    }

    #[test]
    // Simulate many threads which try to write to and then read the data from the cache. We lock
    // the data for each single write and read operation
    fn multi_thread_access() {
        let dir = tempdir().unwrap();
        println!("[multi] temp dir path: {:?}", dir.path());

        let cache = Cache::new(dir.path().join("multi-thread.dat")).expect("MMap allocation");
        let data = get_random_items(100000);

        data.par_iter().for_each(|v| {
            cache.write(v.0, v.1).expect("Data is written to Mmap");
        });

        data.par_iter()
            .for_each(|v| assert_eq!(v.1, cache.read(v.0).expect("Data is read from MMap")));
    }

    // helper function to generate the list of random number from 0..u64::MAX
    fn get_random_items(items: usize) -> Vec<(usize, u64)> {
        let mut rng = rand::thread_rng();
        let mut col = (0..items)
            .map(|_| rng.gen_range(0..u64::MAX))
            .enumerate()
            .collect::<Vec<_>>();
        col.shuffle(&mut rng);
        col
    }

    #[cfg(feature = "nightly")]
    use test::Bencher;

    #[cfg(feature = "nightly")]
    #[bench]
    // run the bench on write and lock is taken for each operation
    fn bench_write(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[writer] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-write.dat")).unwrap();
        let input_data = get_random_items(1000000);

        b.iter(|| {
            input_data.par_iter().for_each(|v| {
                cache.write(v.0, v.1).unwrap();
            });
        })
    }

    #[cfg(feature = "nightly")]
    #[bench]
    // run the bench on read and lock is taken for each operation
    fn bench_read(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[reader] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-read.dat")).unwrap();
        let input_data = get_random_items(1000000);

        cache
            .with_writer(|w| {
                for e in input_data.iter() {
                    w.write(e.0, e.1).unwrap();
                }
                Ok(())
            })
            .unwrap();

        b.iter(|| {
            input_data.par_iter().for_each(|v| {
                let _ = cache.read(v.0).unwrap();
            });
        })
    }

    #[cfg(feature = "nightly")]
    #[bench]
    // run the bench on read and we try to partition the data into chunks of 1000 elements, which
    // are read from the cache while the lock on the data is held
    fn bench_read_batch(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[reader-batch] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-read-batch.dat")).unwrap();
        let input_data = get_random_items(1000000);

        cache
            .with_writer(|w| {
                for e in input_data.iter() {
                    w.write(e.0, e.1).unwrap();
                }
                Ok(())
            })
            .unwrap();

        b.iter(|| {
            // partition our data into chunks with 1000 elements each
            // and read them up from mmap file
            input_data.par_chunks(1000).for_each(|chunk| {
                cache
                    .with_reader(|r| {
                        for e in chunk {
                            let _ = r.read(e.0).unwrap();
                        }
                        Ok(())
                    })
                    .unwrap(); // we must unwrap to make sure the function actually is executed
            });
        })
    }

    #[cfg(feature = "nightly")]
    #[bench]
    // run the bench on write and we try to partition the data into chunks of 1000 elements, which
    // are written in batches to the cache while the lock is held
    fn bench_write_batch(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[write-batch] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-write-batch.dat")).unwrap();
        let input_data = get_random_items(1000000);

        b.iter(|| {
            // partition our data into chunks with 1000 elements each
            // and write them to the mmap file
            input_data.par_chunks(1000).for_each(|chunk| {
                cache
                    .with_writer(|w| {
                        for e in chunk {
                            w.write(e.0, e.1).unwrap();
                        }
                        Ok(())
                    })
                    .unwrap(); // we must unwrap to make sure the function actually is executed
            });
        })
    }
}
