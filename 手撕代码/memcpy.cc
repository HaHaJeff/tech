#include <iostream>
#include <string.h>
#include <assert.h>
void* Memcpy(const void* src, void* dst, size_t n) {
    assert (src != nullptr && dst != nullptr);
    const char *psrc = static_cast<const char*>(src);
    char *pdst = static_cast<char*>(dst);
    if (src == dst) return dst;
    if (src < dst && src + n > dst) {
        pdst += n-1;
        psrc += n-1;
        while (n--) *pdst-- = *psrc--;

    } else {
        while (n--) *pdst++ = *psrc++;

    }
    return dst;
}

int main()
{
    int arr[10] = {1,2,3,4,5,6};
    int* dst = arr + 3;
    Memcpy(arr, dst, 6*sizeof(int));
    for (int i = 0; i < 6; i++) {
        std::cout << dst[i] << std::endl;
    }
}
