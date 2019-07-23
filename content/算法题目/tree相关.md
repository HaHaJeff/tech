# valid binary search tree
**思路：采用递归的方式，root->left->val < root->val && root->right->val > root->val**
**但是这种思路有问题，因为只考虑了局部，而没有考虑全局**
``` cpp
class Solution1 {
    public:
        // 方法有问题：不能只是采用左子节点小于父节点或右子节点大于父节点
        bool isValidBST(TreeNode* root) {
            if (root == nullptr) return true;
            bool left = root->left == nullptr ? true : (root->left->val < root->val ? isValidBST(root->left) : false);
            bool right = root->right == nullptr ? true : (root->right->val > root->val ? isValidBST(root->right) : false);
            return left && right;
        }
};
```

**思路：**
- 左左子树：所有节点都要小于根节点
- 左右子树：所有节点都要大于当前根节点，但是要小于根节点
- 右右子树：所有节点都要大于根节点
- 右左子树：所有节点都要小于当前根节点，但是要大于根节点
``` cpp
class Solution {
    public:
        bool isValidBST(TreeNode* root) {
            return dfs(root, nullptr, nullptr);
        }

        bool dfs(TreeNode* root, TreeNode* minNode, TreeNode* maxNode) {
            if(root == nullptr) return true;

            if(minNode != nullptr) {
                if (root->val <= minNode->val) return false;
            }
            if(maxNode != nullptr) {
                if (root->val >= maxNode->val) return false;
            }
            return dfs(root->left, minNode,root) && dfs(root->right, root, maxNode);
        }
};
```

# two sum iv input as a bst
    **在一个平衡二叉树中找到两数之和为num**
- 方法1：采用递归的是， O(nh)，h最好情况为lgn，最坏为n，额外空间复杂度O(h)
    **思路：采用递归的方式对每一个节点都进行判断，每次判断相当于在整颗数中寻找满足val == k-cur->val（寻找的过程中可以利用bst特性加速）**
    ``` cpp
    class Solution {
        public:
            bool findTarget(TreeNode* root, int k) {
                return dfs(root, root, k);
            }
            bool dfs(TreeNode* root, TreeNode* cur, int val) {
                if (cur == nullptr) return false;
                return search(root, cur, val - cur->val) || dfs(root, cur->left, val) || dfs(root, cur->right, val);
            }
            bool search(TreeNode* root, TreeNode* cur, int val) {
                if (root == nullptr) return false;
                return root->val == val && root != cur || (root->val > val && search(root->left, cur, val)) || (root->val < val && search(root->right, cur, val));
            }
    };
    ```
- 方法2：采用中序遍历+双指针查找 O(n)，额外空间复杂度O(n)
    ``` cpp
    class Solution {
        public:
            bool findTarget(TreeNode* root, int k) {
                stack<TreeNode*> s;
                TreeNode* cur = root;
                vector<int> v;
                while (cur || !s.empty()) {
                    while(cur) {
                        s.push(cur);
                        cur = cur->left;
                    }
                    if (!s.empty()) {
                        cur = s.top();
                        s.pop();
                        v.push_back(cur->val);
                        cur = cur->right;
                    }
                }
                int i = 0, j = v.size()-1;
                while (i < j) {
                    if (v[i] + v[j] == k) return true;
                    v[i] + v[j] < k ? i++ : j--;
                }
                return false;
            }
    };
    ```
- 方法3：采用set对已经遍历过的值进行存储，方法具有普适性，时间复杂度O(n)，空间复杂度O(n)
    ``` cpp
    class Solution {
        public:
            bool findTarget(TreeNode* root, int k) {
                unordered_set<int> s;
                return dfs(root, s, k);
            }
            bool dfs(TreeNode* root, unordered_set<int>& s, int k) {
                if (root == nullptr) return false;
                if (s.count(k-root->val)) { return true;}
                s.insert(root->val);
                return dfs(root->left, s, k) || dfs(root->right, s, k);
            }
    };
```

# unique binary search trees

**递归问题，以i为根节点，那么其左子树[1, i-1]，右子树[i+1, end]，同理递归，当sstart > end时，返回1**

``` cpp
class Solution {
    public:
        int numTrees(int n) {
            dp.resize(n+1);
            for (int i = 0; i < n+1; ++i) {
                dp[i].resize(n+1);
            }
            return recur(1, n);
        }

        int recur(int start, int end) {

            if (start > end) return 1;

            int left = 0;
            int right = 0;

            if (dp[start][end] != 0) return dp[start][end];

            for (int i = start; i <= end; i++) {

                left = recur(start, i - 1);
                right = recur(i + 1, end);
                dp[start][end] += (left*right);
            }
            return dp[start][end];
        }

    private:
        std::vector<std::vector<int>> dp;
};

```

# unique binary search trees ii

**同unique binary search trees**

``` cpp
class Solution {
    public:
        vector<TreeNode*> generateTrees(int n) {

            if (n == 0) return vector<TreeNode*>();

            return generateTreesDFS(1, n);
        }

        vector<TreeNode*> generateTreesDFS(int start, int end) {

            vector<TreeNode*> result;

            if (start > end) {
                result.push_back(NULL);
                return result;
            }

            if (start == end) {
                result.push_back(new TreeNode(start));
                return result;
            }

            for (int i = start; i <= end; i++) {
                vector<TreeNode*> leftTree = generateTreesDFS(start, i - 1);
                vector<TreeNode*> rightTree = generateTreesDFS(i+1, end);

                for (int k = 0; k < leftTree.size(); k++) {
                    for (int z = 0; z < rightTree.size(); z++) {
                        TreeNode* root = new TreeNode (i);
                        root->left = leftTree[k];
                        root->right = rightTree[z];
                        result.push_back(root);
                    }
                }
            }

            return result;
        }
};
```

# sum root to leaf numbers
**dfs解决，自顶向下直接计算**
``` cpp
class Solution {
    public:
        int sumNumbers(TreeNode* root) {
            return calcSum(root, 0);
        }

        int calcSum(TreeNode* root, int sum) {
            if (root == nullptr) return 0;
            if (root->left == nullptr && root->right == nullptr) return root->val + sum * 10;
            return calcSum(root->left, sum * 10 + root->val) + calcSum(root->right, sum * 10 + root->val);
        }
};
```

# smallest subtree with all deepest nodes
```
Input: [3,5,1,6,2,0,8,null,null,7,4]
Output: [2,7,4]
```
![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/01/sketch1.png)
``` cpp
class Solution {
    public:
        TreeNode* subtreeWithAllDeepest(TreeNode* root) {
            TreeNode* maxSubtree = nullptr;

            auto ret = subtree(root);

            return ret.second;
        }

        std::pair<int, TreeNode*> subtree(TreeNode* root) {
            if (root == nullptr) {
                return{ 0, nullptr };
            }

            auto left = subtree(root->left);
            auto right = subtree(root->right);

            return{ max(left.first, right.first) + 1, left.first == right.first ? root : left.first > right.first ? left.second : right.second };

        }
};
```

# most frequent subtree sum

```
Examples 1
Input:

5
/  \
        2   -3
        return [2, -3, 4], since all the values happen only once, return all of them in any order.
        ```

        ``` cpp
        class Solution {
            public:
                vector<int> findFrequentTreeSum(TreeNode* root) {
                    vector<int> ret;

                    int maxCnt = INT_MIN;
                    int sum = subtreeSum(root, maxCnt);

                    for (auto &result : results_) {
                        if (result.second == maxCnt) ret.push_back(result.first);
                    }

                    return ret;

                }

                int subtreeSum(TreeNode* root, int& maxCnt) {
                    int sum = 0;
                    if (root != nullptr) {
                        sum += root->val + subtreeSum(root->left, maxCnt) + subtreeSum(root->right, maxCnt);
                        results_[sum]++;
                        maxCnt = std::max(results_[sum], maxCnt);
                    }
                    return sum;

                }

            private:
                std::map<int, int> results_;
        };
```

# maximum width of binary trees
```
Example 1:

Input: 

1
/   \
        3     2
        / \     \  
        5   3     9 

        Output: 4
        Explanation: The maximum width existing in the third level with the length 4 (5,3,null,9).
        ```

        ``` cpp
        class Solution {
            public:
                int widthOfBinaryTree(TreeNode* root) {
                    if (nullptr == root) return 0;
                    vector<pair<TreeNode*, int>> cur, next;
                    int maxDis = 1;
                    cur.push_back(make_pair(root, 1));

                    while (true) {
                        for (auto iter : cur) {
                            if (iter.first->left != nullptr) next.push_back(make_pair(iter.first->left, iter.second * 2));
                            if (iter.first->right != nullptr) next.push_back(make_pair(iter.first->right, iter.second * 2 + 1));
                        }
                        maxDis = std::max(maxDis, cur.back().second - cur.front().second + 1);
                        if (next.empty()) break;
                        cur = next;
                        next.clear();
                    }
                    return maxDis;
                }
        };
```
# convert bst to greater tree
**思路：记录右边子树的sum即可**

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        TreeNode* convertBST(TreeNode* root) {
            int sum = 0;
            dfs(root, sum);
            return root;
        }
        void dfs(TreeNode* root, int& sum) {
            if (root == nullptr) return;

            dfs(root->right, sum);
            sum += root->val;
            root->val = sum;
            dfs(root->left, sum);
            return;
        }
};
```

# diameter of binary tree

**思路：求得每个节点得左右深度，取l+r+1的最大值**

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        int diameterOfBinaryTree(TreeNode* root) {
            ans = 1;
            dfs(root);
            return ans-1;
        }

        int dfs(TreeNode* root) {
            if (root == nullptr) return 0;
            int l = dfs(root->left);
            int r = dfs(root->right);
            ans = max(l+r+1, ans);
            return max(l, r) + 1;
        }

        int ans;
};
```

# range sum of bst

**思路：在bst中求满足某一范围的节点和**

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        int rangeSumBST(TreeNode* root, int L, int R) {
            if (root == nullptr) return 0;
            int left = 0, right = 0;
            if (L <= root->val && R >= root->val) {
                left = rangeSumBST(root->left, L, root->val)+root->val;
                right = rangeSumBST(root->right, root->val, R);
            } else if (R <= root->val) {
                left = rangeSumBST(root->left, L, R);
            } else if (L >= root->val) {
                right = rangeSumBST(root->right, L, R);
            }
            return left + right;
        }
};
```

# recover binary search tree
**思路：因为只有两个节点被替换过，所以中序遍历BST，发现prev >= root即不满组条件**

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        void recoverTree(TreeNode* root) {
            if (root == nullptr) return;
            prev = new TreeNode(INT_MIN);
            first = second = nullptr;
            dfs(root);
            if (first != nullptr && second != nullptr) swap(first->val, second->val);
            return;
        }
        void dfs(TreeNode*root) {
            if (root == nullptr) return;

            dfs(root->left);
            if (first == nullptr && prev->val >= root->val) first = prev;
            if (first != nullptr && prev->val >= root->val) second = root;
            prev = root;
            dfs(root->right);
        }
        TreeNode* first;
        TreeNode* second;
        TreeNode* prev;
};
```

# range sum query mutable

``` cpp
#include <vector>
using namespace std;
struct SegmentTreeNode {
    int start,end,sum;
    SegmentTreeNode* left;
    SegmentTreeNode* right;

    SegmentTreeNode(int s, int e) : start(s), end(e), sum(0), left(nullptr), right(nullptr) {}
};

struct SegmentTree {
    SegmentTreeNode* buildTree(vector<int>& nums, int start, int end);
    int modifyTree(int index, int val) { return modifyTree(index, val, root); }
    int modifyTree(int index, int val, SegmentTreeNode* node);
    int queryTree(int start, int end) { return queryTree(start, end, root); }
    int queryTree(int start, int end, SegmentTreeNode* node);
    ~SegmentTree();
    void deleteNode(SegmentTreeNode* node);
    SegmentTreeNode* root;
};

SegmentTreeNode* SegmentTree::buildTree(vector<int>& nums, int start, int end) {
    if (start > end) return nullptr;
    SegmentTreeNode* node = new SegmentTreeNode(start, end);
    if (start == end) { node->sum = nums[start]; return node;}
    int mid = start + ((node->end - node->start)>>1);
    node->left = buildTree(nums, start, mid);
    node->right = buildTree(nums, mid + 1, end);
    node->sum = node->left->sum + node->right->sum;
    return node;
}

int SegmentTree::modifyTree(int index, int val, SegmentTreeNode* node) {
    if (node == nullptr) return 0;
    int diff = 0;
    if (node->start == node->end && node->start == index) {
        diff = val - node->sum;
        node->sum = val;
        return diff;
    }
    int mid = node->start + ((node->end - node->start)>>1);
    if (index > mid) {
        diff = modifyTree(index, val, node->right);
    }
    else {
        diff = modifyTree(index, val, node->left);
    }
    node->sum += diff;
    return diff;
}

int SegmentTree::queryTree(int start, int end, SegmentTreeNode* node) {
    if (node == nullptr) return 0;
    if (node->start == start && node->end == end) {
        return node->sum;
    }
    int mid = node->start + ((node->end - node->start)>>1);
    if (start > mid) return queryTree(start, end, node->right);
    if (end <= mid) return queryTree(start, end, node->left);
    return queryTree(start, mid, node->left) + queryTree(mid + 1, end, node->right);
}

void SegmentTree::deleteNode(SegmentTreeNode* node) {
    if (node == nullptr) return;
    deleteNode(node->left);
    deleteNode(node->right);
    delete(node);
}

SegmentTree::~SegmentTree() {
    //deleteNode(root);
}

class NumArray {
    public:
        NumArray(vector<int>& nums) {
            tree.root = tree.buildTree(nums, 0, nums.size() - 1);
        }

        void update(int i, int val) {
            tree.modifyTree(i, val);
        }

        int sumRange(int i, int j) {
            return tree.queryTree(i, j);
        }

        SegmentTree tree;
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * obj->update(i,val);
 * int param_2 = obj->sumRange(i,j);
 */
```

# cousins in binary
**思路：BFS遍历**
- 判断当前层节点的左右子节点是否于x，y相同，如果相同，则false
- 判断当前层节点的值是否全部满足x，y，如果满足，返回true
``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        bool isCousins(TreeNode* root, int x, int y) {
            // same depth but have different parents
            queue<TreeNode*> q;
            q.push(root);
            while (!q.empty()) {
                bool xExisting = false;
                bool yExisting = false;
                int s = q.size();

                for (int i = 0; i < s; i++) {
                    TreeNode* cur = q.front(); q.pop();
                    if (cur->left != nullptr && cur->right != nullptr) {
                        if ((cur->left->val == x && cur->right->val == y) ||
                                (cur->left->val == y && cur->right->val == x)) {
                            return false;
                        }
                    }

                    if (cur->val == x) xExisting = true;
                    if (cur->val == y) yExisting = true;

                    if (cur->left != nullptr) {
                        q.push(cur->left);
                    }
                    if (cur->right != nullptr) {
                        q.push(cur->right); 
                    }
                }

                if (xExisting && yExisting) return true;
            }
            return false;
        }
};
```



# minimum distance between bst nodes

**思路：因为是搜索二叉树，所以最小距离一定在父子节点间，采用树的中序遍历即可**

```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        int minDiffInBST(TreeNode* root) {
            if (root->left != nullptr) minDiffInBST(root->left);
            if (pre >= 0) res = min(res, root->val - pre);
            pre = root->val;
            if (root->right != nullptr) minDiffInBST(root->right);
            return res;
        }

        int pre = -1;
        int res = INT_MAX;
};
```

# path sum iii
**暴力，对每个点都算一次**
``` cpp
class Solution {
    public:
        int pathSum(TreeNode* root, int sum) {
            return root == nullptr ? 0 : (calc(root, sum) + pathSum(root->left, sum) + pathSum(root->right, sum));
        }

        int calc(TreeNode* root, int sum) {
            return root == nullptr ? 0 : (root->val == sum) + calc(root->left, sum - root->val) + calc(root->right, sum - root->val);
        }
};
```

**前缀和**
``` cpp
class Solution {
    public:
        int pathSum(TreeNode* root, int sum) {
            std::unordered_map<int, int> preSum;
            preSum[0] = 1;
            return helper(root, sum, 0, preSum);
        }

        int helper(TreeNode* root, int sum, int curSum, std::unordered_map<int, int>& preSum) {
            if (root == nullptr) return 0;
            curSum += root->val;
            int res = preSum[curSum-sum]; 
            preSum[curSum]++;

            res += helper(root->left, sum, curSum, preSum) + helper(root->right, sum, curSum, preSum);
            preSum[curSum]--;
            return res;
        }
};
```

# path sum

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        bool hasPathSum(TreeNode* root, int sum) {
            if (root == nullptr) return false;
            return helper(root, sum);
        }

        bool helper(TreeNode* root, int sum) {
            if (root == nullptr) return false;
            if (root->left == nullptr && root->right == nullptr && sum == root->val) return true;
            return helper(root->left, sum - root->val) || helper(root->right, sum - root->val);
        }
};
```

# path sum ii
**与path sum类似**
``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        vector<vector<int>> pathSum(TreeNode* root, int sum) {
            vector<int> cur;
            vector<vector<int>> ans;
            helper(root, sum, ans, cur);
            return ans;
        }

        void helper(TreeNode* root, int sum, vector<vector<int>>& ans, vector<int>& cur) {
            if (root == nullptr) return;
            cur.push_back(root->val);
            if (root->left == nullptr && root->right == nullptr && sum == root->val) {
                ans.push_back(cur);
            }
            helper(root->left, sum-root->val, ans, cur);
            helper(root->right, sum-root->val, ans, cur);
            cur.pop_back();
        }
};
```

# minimum absolute difference in bst
- burst force，对每一个点都进行一次运算
``` cpp
class Solution {
    public:
        int getMinimumDifference(TreeNode* root) {
            if (root == nullptr) return INT_MAX;
            int minDiff = min(helper(root->left, root->val, INT_MAX), helper(root->right, root->val, INT_MAX));
            return min(minDiff, min(getMinimumDifference(root->left), getMinimumDifference(root->right)));

        }

        int helper(TreeNode* root, int val, int minDiff) {
            if (root == nullptr) return minDiff;
            minDiff = min(minDiff, abs(root->val - val));
            if (val <= root->val) {
                return helper(root->left, val, minDiff);
            } else {
                return helper(root->right, val, minDiff);
            }
        }
};
```

- 中序遍历

``` cpp
class Solution {
    public:
        int getMinimumDifference**(TreeNode* root) {
            if (root == NULL) **return** 0;
            in res = INT_MAX;
            int prev = INT_MAX;
            helper(root, prev, res);
            return res;
        }
    private:
        void helper**(TreeNode* root, **int**& prev, **int**& res) {
            if (root == NULL) **return**;
            helper(root->left, prev, res);
            int diff = abs(root->val - prev);
            if (diff < res) res = diff;
            prev = root->val;
            helper(root->right, prev, res);
        }
};
```

# convert sorted array to binary search tree
**思路：二分+递归**
``` cpp
class Solution {
    public:
        TreeNode* sortedArrayToBST(vector<int>& nums) {
            int n = nums.size();
            return sortedArrayToBST(nums, 0, n);
        }

        TreeNode* sortedArrayToBST(vector<int>& nums, int start, int end) {
            if(start >= end) {
                return NULL;
            }
            int mid = (start + end) / 2;
            TreeNode* root = new TreeNode(nums[mid]);
            root -> left = sortedArrayToBST(nums, start, mid);
            root -> right = sortedArrayToBST(nums, mid + 1, end);
            return root;
        }
};
```

# maximum binary tree
**思路：二叉树的题目多与二分搜索以及递归相关，此题亦不例外**
``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
            return constructMaximumBinaryTree(nums, nums.begin(), nums.end());
        }

        TreeNode* constructMaximumBinaryTree(vector<int>& nums, const vector<int>::iterator& l, const vector<int>::iterator& r) {
            if (l-r >= 0) return nullptr;

            auto m = max_element(l, r);
            TreeNode* root = new TreeNode(*m);
            root->left = constructMaximumBinaryTree(nums, l, m);
            root->right = constructMaximumBinaryTree(nums, m+1, r);
            return root;
        }
};
```

# add and search word data structure desgin
``` cpp
struct TrieNode {
    TrieNode() : leaf(false) { for (int i = 0; i < 26; i++) child[i] = nullptr; }
    TrieNode* child[26];
    bool leaf;
};

struct Trie {

    Trie() {
        root = new TrieNode();
    }

    ~Trie() {
        delete root;
    }

    TrieNode* root;

    void add(const string& word) {
        TrieNode* cur = root;
        for (auto& ch : word) {
            if (cur->child[ch-'a'] == nullptr) {
                cur->child[ch-'a'] = new TrieNode();
            }
            cur = cur->child[ch-'a'];
        }
        cur->leaf = true;
    }

    bool search(const string& word, TrieNode* node) {
        for (int i = 0; i < word.size() && node != nullptr; i++) {
            if (word[i] == '.') {
                TrieNode* tmp = node;
                string str(word.substr(i+1));
                for (int i = 0; i < 26; i++) {
                    node = tmp->child[i];
                    if (search(str, node)) return true; 
                }
                return false;
            } else {
                node = node->child[word[i]-'a'];    
            }
        }
        return node != nullptr && node->leaf == true;
    }
};

class WordDictionary {
    public:
        /** Initialize your data structure here. */
        WordDictionary() {

        }

        /** Adds a word into the data structure. */
        void addWord(string word) {
            trie.add(word);
        }

        /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
        bool search(string word) {
            return trie.search(word, trie.root);
        }

        Trie trie;
};

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary* obj = new WordDictionary();
 * obj->addWord(word);
 * bool param_2 = obj->search(word);
 */
```

# implement trie prefix tree

``` cpp
class Trie {
    public:
        /** Initialize your data structure here. */
        Trie() {
            root = new TrieNode();
        }

        ~Trie() {
            delete root;
        }

        /** Inserts a word into the trie. */
        void insert(string word) {
            TrieNode* node = root;

            for (auto& ch : word) {
                if (node->child[ch-'a'] == nullptr) {
                    node->child[ch-'a'] = new TrieNode();
                }
                node = node->child[ch-'a'];
            }

            node->leaf = true;
        }

        /** Returns if the word is in the trie. */
        bool search(string word) {
            TrieNode* node = root;

            for (int i = 0; i < word.size() && node != nullptr; i++) {
                node = node->child[word[i] - 'a'];
            }

            return node != nullptr && node->leaf == true;
        }

        /** Returns if there is any word in the trie that starts with the given prefix. */
        bool startsWith(string prefix) {

            TrieNode* node = root;
            for (int i = 0; i < prefix.size() && node != nullptr; i++) {
                node = node->child[prefix[i] - 'a'];
            }
            return node != nullptr;
        }

        struct TrieNode {
            TrieNode* child[26];
            bool leaf;
            TrieNode() {
                leaf = false;
                for (int i = 0; i < 26; i++) {
                    child[i] = nullptr;
                }
            }
        };

        TrieNode* root;
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
```

# distribute coins in binary tree

**思路：二叉树分金币问题**

- 深度优先遍历所有的节点
- 每一个节点返回其子节点多余的或者少的金币

``` cpp
class Solution {
    public:
        int distributeCoins(TreeNode* root) {
            int ans = 0;
            helper(root, ans);
            return ans;
        }

        int helper(TreeNode* root, int& ans) {
            if (root == nullptr) return 0;
            int l = helper(root->left, ans);
            int r = helper(root->right, ans);
            ans += abs(l) + abs(r);
            return root->val + l + r - 1;
        }
};
```

# leaf similar trees

**思路：dfs遍历，使用string记录叶子节点**

``` cpp
class Solution {
    public:
        bool leafSimilar(TreeNode* root1, TreeNode* root2) {
            string str1; string str2;
            dfs(root1, str1); dfs(root2, str2);
            return str1 == str2;
        }

        void dfs(TreeNode* root, string& str) {
            if (root == nullptr) return;
            if (root->left == nullptr && root->right == nullptr) str += to_string(root->val) + "#";
            dfs(root->left, str); dfs(root->right, str);
        }
};
```

# trim a binary search tree

**思路：挑选满足[L, R]的树节点，递归解决**

``` cpp
class Solution {
    public:
        TreeNode* trimBST(TreeNode* root, int L, int R) {
            if (root == nullptr) return nullptr;
            if (root->val < L) return trimBST(root->right, L, R);
            if (root->val > R) return trimBST(root->left, L, R);

            root->left = trimBST(root->left, L, R);
            root->right = trimBST(root->right, L, R);
            return root;
        }
};
```

# find mode in binary search tree

- 首先找到最大频率(中序遍历)
    - 然后再次中序遍历输出==最大频率的node

    ``` cpp
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    class Solution {
        public:
            vector<int> findMode(TreeNode* root) {
                int maxFreq = 0, prev1 = INT_MIN, curFreq1 = 0;
                getFreq(root, maxFreq, prev1, curFreq1);
                int prev2 = INT_MIN, curFreq2 = 0;
                getModes(root, maxFreq, prev2, curFreq2);

                return modes;
            }

            void getFreq(TreeNode* root, int& maxFreq, int& prev, int& curFreq) {
                if (root == nullptr) return;
                getFreq(root->left, maxFreq, prev, curFreq);

                ++(curFreq*=(prev == root->val ? 1 : 0));
                prev = root->val;
                maxFreq = max(maxFreq, curFreq);

                getFreq(root->right, maxFreq, prev, curFreq);
            }

            void getModes(TreeNode* root, int maxFreq, int& prev, int& curFreq) {
                if (root == nullptr) return;
                getModes(root->left, maxFreq, prev, curFreq);

                ++(curFreq*=(prev == root->val ? 1 : 0));
                prev = root->val;

                if (curFreq == maxFreq) {
                    modes.push_back(root->val);
                }

                getModes(root->right, maxFreq, prev, curFreq);
            }

            vector<int> modes;
    };
```
# binary tree maximum path sum
**思路：递归遍历，如果sum < 0，则舍弃**
``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    public:
        int maxPathSum(TreeNode* root) {
            sum = INT_MIN;
            helper(root);
            return sum;
        }

        int helper(TreeNode* root) {
            if (root == nullptr) return 0;
            int left = max(0, helper(root->left));
            int right = max(0, helper(root->right));
            sum = max(sum, left+right+root->val); 
            return max(left, right) + root->val;
        }

        int sum;
};
```
