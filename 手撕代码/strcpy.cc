#include <iostream>
#include <assert.h>
#include <string.h>

char* Strcpy(const char* src, char* dst) {
    if (src == dst) return dst;
    assert(dst != nullptr && src != nullptr);
    int s1 = strlen(src)+1;
    if (src < dst && src + s1 > dst) {
        dst += s1-1;
        src += s1-1;
        while (s1--) {
            *dst-- = *src--;
        }

    } else {
        while (s1--) {
            *dst++ = *src++;

        }

    }
    return dst;

}

int main()
{
    char src[10] = {'1', '2', '3', '4', '5', '\0'};
    char*dst = src + 2;
    Strcpy(src, dst);
    std::cout << src << std::endl;
}
