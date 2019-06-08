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