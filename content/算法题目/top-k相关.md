# top k frequent elements

- **使用partition方完成top-k**
``` cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> freq;
        auto it = nums.begin();
        for (int n : nums) if (!freq[n]++) *it++ = n;
        nums.resize(freq.size());
        nth_element(
            nums.begin(), nums.begin() + (k - 1), nums.end(),
            [&](int a, int b) { return freq[a] > freq[b]; });
        nums.resize(k);
        return move(nums);
    }
};
```

- **使用最大堆完成top-k**
``` cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> freq;
        for (int n : nums) freq[n]++;
        
        priority_queue<pair<int,int>> q;
        
        for (auto& f : freq) q.push({f.second, f.first});
        
        vector<int> ans;
        
        for (int i = 0; i < k && !q.empty(); i++) {
            ans.push_back(q.top().second);
            q.pop();
        }
        
        return ans;
    }
};
```

- **使用最小堆完成top-k**
``` cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> freq;
        for (int n : nums) freq[n]++;
        
        priority_queue<pair<int,int>> q;
        
        for (auto& f : freq) q.push({f.second, f.first});
        
        vector<int> ans;
        
        for (int i = 0; i < k && !q.empty(); i++) {
            ans.push_back(q.top().second);
            q.pop();
        }
        return ans;
    }
};
```

# top k frequent words

**最大堆解决**
``` cpp
class Solution {
public:
    vector<string> topKFrequent(vector<string>& words, int k) {
        std::unordered_map<string, int> countMap;
        for(auto& word : words) {
            countMap[word]++;
        }
        std::vector<pair<string, int>> countVec(countMap.begin(), countMap.end());
        auto comp = [](const pair<string, int>& lhs, const pair<string, int>& rhs) {
            return lhs.second > rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
        };
        std::priority_queue<pair<string, int>, vector<pair<string, int>>, decltype(comp)> q(comp);
        for (auto& item : countVec) {
            q.push(item);
            if (q.size() > k) q.pop();
        }
        std::vector<string> output;
        while(!q.empty()) {
            output.insert(output.begin(), q.top().first);
            q.pop();
        }
        return output;
    }
};
```

# k closest

- **最大堆**
``` cpp
class Solution {
public:
    double distance(vector<int>& point) {
        return sqrt(point[0]*point[0] + point[1]*point[1]);
    }
    
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        vector<vector<int>> ans;
        
        auto comp = [&](vector<int>& lhs, vector<int>& rhs) {return distance(lhs) < distance(rhs);};
        priority_queue<vector<int>, vector<vector<int>>, decltype(comp)> q(comp);
        
        for (auto& point : points) {
            q.push(point);
            
            if (q.size() > K) {
                q.pop();
            }
        }
        
        while (!q.empty()){
            ans.push_back(q.top());
            q.pop();
        }
        
        return ans;
    }
};
```

- **nth_element**
``` cpp
class Solution {
public:
    double distance(vector<int>& point) {
        return point[0]*point[0] + point[1]*point[1];
    }
    
    int partition(vector<vector<int>>& points, int start, int end) {
        auto pivot = points[start];
        
        while (start < end) {
            while (distance(points[end]) >= distance(pivot) && start < end) --end;
            points[start] = points[end];
            while (distance(points[start]) <= distance(pivot) && start < end) ++start;
            points[end] = points[start];
        }
        points[start] = pivot;
        return start;
    }
    
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {

        int m  = 0;
        int start = 0, end = points.size() - 1;
        while (1) {
            if ((m = partition(points, start, end)) == K-1) break;
            else if (m < K-1) start = m+1;
            else end = m-1;
            
        }     
        vector<vector<int>> ans(points.begin(), points.begin()+K);
        return ans;
    }
};
```

# partition算法

``` cpp
int partition(vector<int>& nums, int left, int right) {
    int pivot = nums[left];
    
    while(left < right) {
        while (nums[right] >= pviot && left < right) --right;
    	nums[left++] = nums[right];
    	while (nums[left] <= pviot && left < right) ++left;
    	nums[right--] = nums[left];
    }
    nums[left] = pviot;
    return left;
}
```

# 堆算法

**最大堆**
``` cpp
void heapify(vector<int>& nums, int start, int end) {
    int parent = start, child = 2*parent+1;
    int pivot = nums[parent];
    while (child < end) {
        child = (child+1 < end && nums[child] < nums[child+1] ? nums[child+1] : nums[child]);
        if (nums[parent] < nums[child]) break;
        nums[parent] = nums[child];
        parent = child;
        child = 2*parent+1;
    }
    nums[child] = pivot;
} 
```