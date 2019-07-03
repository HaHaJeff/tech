# satisfiablity of equality equations
- 判断等式是否满足

``` cpp
Input: ["a==b","b!=a"]
Output: false
Explanation: If we assign say, a = 1 and b = 1, then the first equation is satisfied, but not the second.  There is no way to assign the variables to satisfy both equations.

Input: ["b==a","a==b"]
Output: true
Explanation: We could assign a = 1 and b = 1 to satisfy both equations.

Input: ["a==b","b==c","a==c"]
Output: true

Input: ["a==b","b==c","a==c"]
Output: true

```
**采用并查集的方式对等式进行调整**
``` cpp
class Solution {
public:
    bool equationsPossible(vector<string>& equations) {
        vector<int> uf(26, 0);
        for (int i = 0; i < 26; i++) uf[i] = i;
        for (auto& str : equations) {
            if (str[1] == '=') {
                uf[find(str[0]-'a', uf)] = find(str[3]-'a', uf);
            }
        }
        
        for (auto& str : equations) {
            if (str[1] == '!' && find(str[0]-'a', uf) == find(str[3]-'a', uf)) {
                return false;
            }
        }
        return true;
    }
    
    int find(int a, vector<int>& uf) {
        if (a != uf[a]) uf[a] = find(uf[a], uf);
        return uf[a];
    }
};
```

# evaluate division

``` cpp
class Solution {
public:
    struct Node {
        string root;
        double val;
        Node() {}
        Node(string root, double val) : root(root), val(val) {}
    };
    unordered_map<string, Node> u;
    unordered_map<string, int> rank;
    string findRoot(string x) {
        while(u[x].root != x) {
            u[x].val = u[x].val * u[ u[x].root ].val;
            u[x].root = u[ u[x].root ].root;
            x = u[x].root;
        }
        return x;
    }
    void unionRoot(string x, string y, double val) {
        string i = findRoot(x), j = findRoot(y);
        if(i != j) {
            if(rank[i] >= rank[j]) {
                u[j].root = i;
                u[j].val = u[x].val / u[y].val / val;
                rank[i] += rank[j];
            }
            else {
                u[i].root = j;
                u[i].val = u[y].val / u[x].val * val;
                rank[j] += rank[i];
            }
        }
    }
    
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        for(int i = 0; i < equations.size(); i++) {
            string from = equations[i][0], to = equations[i][1];
            double val = values[i];
            if(!u.count(from)) {
                u[from] = Node(from, 1);
                rank[from] = 1;
            }
            if(!u.count(to)) {
                u[to] = Node(to, 1);
                rank[to] = 1;
            }
            unionRoot(from, to, val);
        }
        vector<double> res(queries.size());
        for(int i = 0; i < queries.size(); i++) {
            string from = queries[i][0], to = queries[i][1];
            if(!u.count(from) || !u.count(to) || findRoot(from) != findRoot(to)) {
                res[i] = -1;
                continue;
            }
            res[i] = u[from].val / u[to].val;
        }
        return res;
    }
};
```