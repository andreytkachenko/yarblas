use std::alloc;

pub struct Alloc {
    ptr: *mut u8,
    size: usize,
    layout: alloc::Layout,
}

impl Alloc {
    pub fn new(size: usize) -> Alloc {
        const ALIGN: usize = 32;
        let layout = alloc::Layout::from_size_align(size, ALIGN).unwrap();
        let ptr = unsafe { alloc::alloc(layout) };
        Alloc { ptr, size, layout }
    }

    pub fn ptr<F>(&self) -> *mut F {
        self.ptr as *mut F
    }

    pub unsafe fn into_vec<F>(self) -> Vec<F> {
        let item_size = std::mem::size_of::<F>();
        let item_count = self.size / item_size;
        let ptr = self.ptr as *mut F;
        std::mem::forget(self);

        Vec::from_raw_parts(ptr, item_count, item_count)
    }
}

impl Drop for Alloc {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr, self.layout);
        }
    }
}
