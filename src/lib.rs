#![cfg_attr(all(test, feature = "nightly"), feature(test))]

#[cfg(all(test, feature = "nightly"))]
extern crate test;

use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use memmap2;

type Result<T, E = Box<dyn std::error::Error + Sync + Send>> = std::result::Result<T, E>;

#[derive(Debug, Clone)]
pub struct Cache {
    inner: Arc<Mutex<CacheInner>>,
}

const PAGE_SIZE: usize = 8096;

impl Cache {
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
    fn get_location(index: usize) -> usize {
        (index << 3) as usize
    }

    fn with_reader<F>(&self, f: F) -> Result<u64>
    where
        F: Fn(&dyn CacheReader) -> Result<u64>,
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

    fn with_writer<F>(&self, f: F) -> Result<()>
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

    pub fn write(&self, index: usize, value: u64) -> Result<()> {
        self.with_writer(|writer| writer.write(index, value))
    }

    pub fn read(&self, index: usize) -> Result<u64> {
        self.with_reader(|reader| reader.read(index))
    }

    pub fn flsuh(&self) -> Result<()> {
        self.with_writer(|writer| writer.flush())
    }
}

#[derive(Debug)]
pub struct CacheInner {
    file: File,
    elements_count: usize,
    memmap: memmap2::MmapMut,
}

pub trait CacheReader {
    fn read(&self, index: usize) -> Result<u64>;
}

impl CacheReader for CacheInner {
    fn read(&self, index: usize) -> Result<u64> {
        let mut buf: [u8; 8] = [0; 8];

        // TODO: check if we access undefined indicies
        unsafe {
            self.memmap
                .as_ptr()
                .add(Cache::get_location(index))
                .copy_to(buf.as_mut_ptr(), std::mem::size_of::<u64>())
        };
        Ok(u64::from_le_bytes(buf))
    }
}

pub trait CacheWriter {
    fn write(&mut self, index: usize, value: u64) -> Result<()>;
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
    use rand::Rng;
    use rayon::prelude::*;
    use tempfile::tempdir;

    #[test]
    fn single_threaded() {
        // data to write into the cache
        let data = get_random_items(100000);

        // set up our cache file
        let dir = tempdir().unwrap();
        println!("[single] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("single-thread.dat")).expect("Mmap allocation");

        for (i, d) in data.iter().enumerate() {
            cache.write(i, *d as u64).expect("Data is written to MMap");
        }

        for (i, d) in data.iter().enumerate() {
            let data_from_mmap: u64 = cache.read(i).expect("Data is read from MMap");
            assert_eq!(data_from_mmap, *d as u64);
        }
    }

    #[test]
    fn multi_thread_access() {
        let dir = tempdir().unwrap();
        println!("[multi] temp dir path: {:?}", dir.path());

        let cache = Cache::new(dir.path().join("multi-thread.dat")).expect("MMap allocation");
        let data = get_random_items(100000);

        data.par_iter().enumerate().for_each(|(i, v)| {
            cache.write(i, *v as u64).expect("Data is written to Mmap");
        });

        data.par_iter().enumerate().for_each(|(i, v)| {
            assert_eq!(*v as u64, cache.read(i).expect("Data is read from MMap"))
        });
    }

    // helper function to generate the list of random number from 0..u64::MAX
    fn get_random_items(items: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        (0..items)
            .map(|_| rng.gen_range(0..u64::MAX as usize))
            .collect()
    }

    #[cfg(feature = "nightly")]
    use test::Bencher;

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_writer(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[writer] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-write.dat")).unwrap();
        let input_data: Vec<usize> = get_random_items(1000000);

        b.iter(|| {
            input_data.par_iter().enumerate().for_each(|(i, v)| {
                cache.write(i, *v as u64).unwrap();
            });
        })
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_read(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[reader] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-read.dat")).unwrap();
        let input_data: Vec<usize> = get_random_items(1000000);

        input_data.par_iter().enumerate().for_each(|(i, v)| {
            cache.write(i, *v as u64).unwrap();
        });
        cache.flsuh().unwrap();

        b.iter(|| {
            input_data.par_iter().enumerate().for_each(|(i, _)| {
                let _ = cache.read(i).unwrap();
            });
        })
    }
}
