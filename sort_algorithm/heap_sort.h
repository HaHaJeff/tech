#include <vector>
using namespace std;

void HeapSort(vector<int>& nums);
void HeapSortCore(vector<int>& nums, int len);
void MaxHeapify(vector<int>& nums, int start, int end);

void HeapSort(vector<int>& nums) {
    HeapSortCore(nums, nums.size());
}

void HeapSortCore(vector<int>& nums, int len) {
    for (int i = len/2-1; i >= 0; i--) {
        MaxHeapify(nums, i, len-1);
    }

    for (int i = len-1; i > 0; i--) {
        swap(nums[i], nums[0]);
        MaxHeapify(nums, 0, i-1);
    }
}

void MaxHeapify(vector<int>& nums, int start, int end) {
    int parent = start, son = 2*parent+1;
    while (son <= end) {
        son = (son+1<=end && nums[son] < nums[son+1])?son+1:son;
        if (nums[parent] > nums[son])
            break;
        swap(nums[parent], nums[son]);
        parent = son;
        son = 2 * parent+1;
    }
}
