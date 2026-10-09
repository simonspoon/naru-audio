//! Fallbacks for `__std_find_first_of_trivial_pos_{1,2}` (see build.rs).
//! Same contract as the STL's: the index in `first1` of the first element
//! that also occurs in `first2`, or `usize::MAX` when there is none.

/// # Safety
/// Both ranges must be readable for the given element counts.
unsafe fn find_first_of<T: Copy + PartialEq>(
    first1: *const T,
    count1: usize,
    first2: *const T,
    count2: usize,
) -> usize {
    let (hay, set) = unsafe {
        (
            std::slice::from_raw_parts(first1, count1),
            std::slice::from_raw_parts(first2, count2),
        )
    };
    hay.iter()
        .position(|c| set.contains(c))
        .unwrap_or(usize::MAX)
}

/// # Safety
/// See [`find_first_of`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn naru_find_first_of_trivial_pos_1(
    first1: *const u8,
    count1: usize,
    first2: *const u8,
    count2: usize,
) -> usize {
    unsafe { find_first_of(first1, count1, first2, count2) }
}

/// # Safety
/// See [`find_first_of`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn naru_find_first_of_trivial_pos_2(
    first1: *const u16,
    count1: usize,
    first2: *const u16,
    count2: usize,
) -> usize {
    unsafe { find_first_of(first1, count1, first2, count2) }
}
