# Mem Stash

This is a test Project, which inspired by https://github.com/nyurik/osm-node-cache (as of [`71e9965`](https://github.com/nyurik/osm-node-cache/tree/71e9965d64182feb4375b59f11e862704425adf2))

It's a simple re-implementation of the mmap backed cache for u64's, using the Mutex to provide the multi-threaded access to the content.


**Note**: Current implementation does not have hard limits on the size of mmap and underliying file and it can grow indefenitely.
It's up to the user of this lib to stay on the safe side.
