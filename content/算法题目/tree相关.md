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