#include <vector>
#include <stack>

using namespace std;

void MergeSort(vector<int>& nums);
void MSortCoreRec(vector<int>& nums, int start, int end);
void MSortCoreIter(vector<int>& nums, int start, int end);

void MergeSort(vector<int>& nums) {
    MSortCoreIter(nums, 0, nums.size() - 1); 
}

void MSortCoreRec(vector<int>& nums, int start, int end) {
    if (start >= end) return;

    int len = end - start, mid = (len >> 1) + start;
    int s1 = start, e1 = mid, s2 = mid+1, e2 = end;
    MSortCoreRec(nums, s1, e1);
    MSortCoreRec(nums, s2, e2);

    vector<int> tmp(end - start + 1);
    int i = 0;
    while (s1 <= e1 && s2 <= e2)
        tmp[i++] = nums[s1] < nums[s2] ? nums[s1++] : nums[s2++];
    while (s1 <= e1)
        tmp[i++] = nums[s1++];
    while (s2 <= e2)
        tmp[i++] = nums[s2++];
    copy(tmp.begin(), tmp.end(), nums.begin()+start);
}

void MSortCoreIter(vector<int>& nums, int start, int end) {
    int len = nums.size();
    vector<int> tmp(len);
    for (int seg = 1; seg < len; seg++) {
        for (int s = 0; s < len; s += (seg + seg)) {
            int low = s, mid = min(len, s + seg), high = min(len, s + seg + seg);
            int s1 = low, e1 = mid, s2 = mid, e2 = high;
            int i = s;
            while(s1 < e1 && s2 < e2)
                tmp[i++] = nums[s1] < nums[s2] ? nums[s1++] : nums[s2++];
            while (s1 < e1)
                tmp[i++] = nums[s1++];
            while (s2 < e2)
                tmp[i++] = nums[s2++];
        }
        copy(tmp.begin(), tmp.end(), nums.begin());
    }
}
