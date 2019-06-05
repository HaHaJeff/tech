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