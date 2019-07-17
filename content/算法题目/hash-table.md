# contiguous array

**思路：利用hash table存储，0=-1， 1=1，累加和，当发现hash table中已经包含过当前sum时，说明从hash table中之前sum出现的位置到现在sum的位置中0，1出现相同**

``` cpp
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        unordered_map<int, int> m;
        int cnt = 0, ans = 0;
        m[0] = -1;
        for (int i = 0; i < nums.size(); i++) {
            cnt = cnt + (nums[i] == 1 ? 1 : -1);
            if (m.count(cnt)) ans = max(ans, i - m[cnt]);
            else m[cnt] = i;
        }
        return ans;
    }
};
```