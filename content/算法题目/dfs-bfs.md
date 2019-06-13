# shortest-bridge
- 使用深度优先遍历将某一块大陆标记出来，采用队列对大陆覆盖的块进行存储
- 使用广度优先比例对队列进行计算
``` cpp
class Solution {
public:
	int shortestBridge(vector<vector<int>>& A) {

		queue<pair<int, int>> q;
		int row = A.size(), col = A[0].size();
		visited.resize(row, vector<int>(col, 0));
		bool marked = false;
		for (int i = 0; i < row; i++) {
			if (marked == true) break;
			for (int j = 0; j < col; j++) {
				if (A[i][j] == 1) {
					dfs(i, j, A, q);
					marked = true;
					break;
				}
			}
		}

		return bfs(A, q);
	}

	// 深度优先遍历负责区分两块大陆
	void dfs(int i, int j, vector<vector<int>>& A, queue<pair<int, int>>& q) {
		if (i < 0 || i >= A.size() || j < 0 || j >= A[0].size() || visited[i][j] == 1 || A[i][j] == 0) return;

		visited[i][j] = 1;
		q.push({ i, j });

		dfs(i + 1, j, A, q);
		dfs(i - 1, j, A, q);
		dfs(i, j + 1, A, q);
		dfs(i, j - 1, A, q);
	}

	// 广度优先遍历用于计算连接两块大陆的最小bridge数目
	int bfs(vector<vector<int>>& A, queue<pair<int, int>>& q) {
		int bridges = 0;
		vector<pair<int, int>> dir = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
		while (!q.empty()) {
			int s = q.size();
			for (int i = 0; i < s; i++) {
                pair<int, int> point = q.front(); q.pop();
				for (auto& d : dir) {
					int curX = point.first + d.first, curY = point.second + d.second;
					if (curX >= 0 && curX < A.size() && curY >= 0 && curY < A[0].size() && visited[curX][curY] == 0) {
                        if (A[curX][curY] == 1) return bridges;
						visited[curX][curY] = 1;
                        q.push({ curX, curY });
						
					}
				}

			}

			bridges++;
		}
		return bridges;
	}

	vector<vector<int>> visited;
};
```



# pacific-atlantic-water-flow

每个坐标点表示该点的水位高度，只能由高水位流向低水位，求哪些点可以同时流向太平洋和大西洋

```
Given the following 5x5 matrix:

  Pacific ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * Atlantic

Return:

[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (positions with parentheses in above matrix).
```

```
class Solution {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
        if(matrix.size() == 0 || matrix[0].size() == 0) return {};
        int m = matrix.size();
        int n = matrix[0].size();
        
        dir.push_back({0, 1}); dir.push_back({0, -1}); dir.push_back({1, 0}); dir.push_back({-1, 0}); 
        vector<vector<int>> pacific(m, vector<int>(n, 0));
        vector<vector<int>> atlantic(m, vector<int>(n, 0));
        queue<pair<int, int>> pQueue;
        queue<pair<int, int>> aQueue;
        
        for (int i = 0; i < m; i++) {
            pQueue.push({i, 0});
            aQueue.push({i, n-1});
            pacific[i][0] = 1;
            atlantic[i][n-1] = 1;
        }
        for (int j = 0; j < n; j++) {
            pQueue.push({0, j});
            aQueue.push({m-1,j});
            pacific[0][j] = 1;
            atlantic[m-1][j] = 1;            
        }
        bfs(matrix, pQueue, pacific);
        bfs(matrix, aQueue, atlantic);
        vector<vector<int>> ans;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (pacific[i][j] == 1 && atlantic[i][j] == 1) ans.push_back({i, j});
            }
        }
        return ans;
    }
    
    void bfs(vector<vector<int>>& matrix, queue<pair<int, int>>& q, vector<vector<int>>& visited) {
        int m = matrix.size(); int n = matrix[0].size();
        while (!q.empty()) {
            pair<int, int> t = q.front(); q.pop();
            int curX = t.first; int curY = t.second;
            
            for (auto& d : dir) {
                int x = curX + d.first; int y = curY + d.second;
                if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y] == 1 || matrix[x][y] < matrix[curX][curY]) continue;
                visited[x][y] = 1;
                q.push({x, y});
            }
        }
    }
    
    vector<pair<int, int>> dir;
};
```

# course schedule ii

**思路：采用深度优先遍历图即可**

- 边表示课程之间的偏序关系，例如A-B表示学习B之前需要先学习A
- 深度优先遍历该图，当遍历到最后一个节点时，表示学习该节点表示的课程之前，需要学习遍历路径上所有的课程
- 需要注意是否存在环，采用todo以及done判断是否有环，如果在深度优先遍历中出现todo == false && done == true，则表示遇见环了

``` cpp
class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses);
        for (auto& p : prerequisites) {
            graph[p[1]].push_back(p[0]);
        }
        vector<bool> todo(numCourses, true), done(numCourses, false);
        vector<int> ans;
        for (int i = 0; i < numCourses; i++) {
            if (done[i] == true) continue;
            if (circum(graph, todo, done, ans, i)) return {};
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
    
    bool circum(const vector<vector<int>>& graph, vector<bool>& todo, vector<bool>& done, vector<int>& ans, int node) {
        if (todo[node] == false && done[node] == true) return true;
        if (done[node] == true) return false;
        todo[node] = false; done[node] = true;
        for (int nei : graph[node]) {
            if (circum(graph, todo, done, ans, nei) == true) return true;
        }
        ans.push_back(node);
        todo[node] = true;
        return false;
    }
    
};
```

# course schedule

``` cpp
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses);
        for (auto& p : prerequisites) {
            graph[p[1]].push_back(p[0]);
        }
        vector<bool> todo(numCourses, true), done(numCourses, false);
        for (int i = 0; i < numCourses; i++) {
            if (done[i] == true) continue;
            if (circum(graph, todo, done, i)) return false;
        }
        return true;
    }
    
    bool circum(const vector<vector<int>>& graph, vector<bool>& todo, vector<bool>& done, int node) {
        if (todo[node] == false && done[node] == true) return true;
        if (done[node] == true) return false;
        todo[node] = false; done[node] = true;
        for (int nei : graph[node]) {
            if (circum(graph, todo, done, nei) == true) return true;
        }
        todo[node] = true;
        return false;
    }
};
```

# keys and rooms

**思路：BFS**

``` cpp
class Solution {
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        stack<int> todo;
        todo.push(0);
        unordered_set<int>done;
        while (!todo.empty()) {
            int cur = todo.top(); todo.pop();
            done.insert(cur);
            for (int i : rooms[cur]) {
                if (done.count(i)) continue;
                todo.push(i);
            }
        }
        return done.size() == rooms.size();
    }
};
```