#include <iostream>
#include <algorithm>
#include <assert.h>
int Atoi(char* str) {
    assert(str != nullptr);
    while (isspace(*str)) ++str;
    bool minus = false;
    if (*str == '+') {
        minus = false;
        ++str;

    } else if (*str == '-') {
        minus = true;
        ++str;

    }
    long long num = 0;
    int flag = minus ? -1 : 1;
    while (*str != '\0' && isdigit(*str)) {
        num = num*10 + flag * (*str-'0');
        if(!minus && num > 0x7FFFFFFF) {
            num = 0;
            break;

        }
        if (minus && num < 0x80000000) {
            num = 0;
            break;

        }
        ++str;

    }
    return num;
}

int main()
{
    int num = Atoi("2147483648");
    std::cout << num << std::endl;
}
