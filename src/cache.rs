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

    pub fn write(&self, index: usize, value: u64) -> Result<()> {
        self.with_writer(|writer| writer.write(index, value))
    }

    pub fn read(&self, index: usize) -> Result<u64> {
        let mut result: Option<u64> = None;
        self.with_reader(|reader| {
            result = reader.read(index).ok();
            Ok(())
        })?;
        result.ok_or_else(|| format!("Unable to find requested index = {}", index).into())
    }

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

pub trait CacheReader {
    fn read(&self, index: usize) -> Result<u64>;
}

impl CacheReader for CacheInner {
    fn read(&self, index: usize) -> Result<u64> {
        let mut buf: [u8; 8] = [0; 8];

        if index > self.elements_count {
            return Err(format!(
                "Unable to access index = {} out of bounds, max available index = {}",
                index, self.elements_count
            )
            .into());
        }

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
    fn bench_read(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[reader] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-read.dat")).unwrap();
        let input_data = get_random_items(1000000);

        input_data.par_iter().for_each(|v| {
            cache.write(v.0, v.1).unwrap();
        });
        cache.flush().unwrap();

        b.iter(|| {
            input_data.par_iter().for_each(|v| {
                let _ = cache.read(v.0).unwrap();
            });
        })
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_read_batch(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[reader-batch] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-read-batch.dat")).unwrap();
        let input_data = get_random_items(1000000);

        input_data.par_iter().for_each(|v| {
            cache.write(v.0, v.1).unwrap();
        });

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
    fn bench_write_batch(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        println!("[write-batch] temp dir path: {:?}", dir.path());
        let cache = Cache::new(dir.path().join("test-cache-write-batch.dat")).unwrap();
        let input_data = get_random_items(1000000);

        input_data.par_iter().for_each(|v| {
            cache.write(v.0, v.1).unwrap();
        });

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
