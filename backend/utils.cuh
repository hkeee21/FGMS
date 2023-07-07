# pragma once

/*******************************************************************
device functions
*/
__device__ __forceinline__ int binary_search(
                            const int *S_csrRowPtr, const int eid, 
                            const int start, const int end) {
    
    int lo = start, hi = end;
    if (lo == hi){
        return lo;
    }
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(S_csrRowPtr + mid) <= eid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (__ldg(S_csrRowPtr + hi) <= eid) {
        return hi;
    } else {
        return hi - 1;
    }
}