#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <stack>
#include <thread>
#include <map>
#include <set>
#include <mutex>
#include <sstream>

using namespace std;

bool veccompare(const vector<int>& vec1, const vector<int>& vec2) {
	if (vec1.size() != 2 || vec2.size() != 2) return false;
	return vec1[0] < vec2[0];
}

vector<vector<int>> merge(vector<vector<int>>& intervals) {
	if (intervals.size() <= 1)return intervals;
	stable_sort(intervals.begin(), intervals.end(), veccompare);
	int left = intervals[0][0], right = intervals[0][1];
	vector<vector<int>> ans;
	for (int i = 1; i < intervals.size(); i++) {
		if (right >= intervals[i][0]) {
			right = max(intervals[i][1], right);
		}
		else {
			ans.push_back(vector<int>({ left,right }));
			left = intervals[i][0];
			right = intervals[i][1];
		}
	}
	ans.push_back(vector<int>({ left,right }));
	return ans;
}

vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
	int left = -1, right = -1;
	for (int i = 0; i < intervals.size(); i++) {
		if (newInterval[0] <= intervals[i][1]) {
			left = i;
			break;
		}
	}
	for (int j = intervals.size() - 1; j >= 0; --j) {
		if (newInterval[1] >= intervals[j][0]) {
			right = j;
			break;
		}
	}
	vector<vector<int>> ans;
	if (left < 0) {
		ans = intervals;
		ans.push_back(newInterval);
		return ans;
	}
	if (right < 0) {
		ans = intervals;
		ans.emplace(ans.begin(), newInterval);
		return ans;
	}
	for (int i = 0; i < left; i++) {
		ans.push_back(intervals[i]);
	}
	if (right < left) {
		ans.push_back(newInterval);
	}
	else {
		int l, r;
		l = min(newInterval[0], min(intervals[left][0], intervals[right][0]));
		r = max(newInterval[1], max(intervals[left][1], intervals[right][1]));
		ans.push_back(vector<int>({ l, r }));
	}

	for (int i = right + 1; i < intervals.size(); i++) {
		ans.push_back(intervals[i]);
	}
	return ans;
}

bool canJump(vector<int>& nums) {
	int rightMost = 0;
	for (int i = 0; i < nums.size(); i++) {
		if (i <= rightMost) {
			rightMost = max(rightMost, i + nums[i]);
			if (rightMost >= nums.size() - 1)return true;
		}
	}
	return false;
}


int getMaxRepetitions(string s1, int n1, string s2, int n2) {
	if (n1 == 0) {
		return 0;
	}
	int s1cnt = 0, index = 0, s2cnt = 0;
	// recall 是我们用来找循环节的变量，它是一个哈希映射
	// 我们如何找循环节？假设我们遍历了 s1cnt 个 s1，此时匹配到了第 s2cnt 个 s2 中的第 index 个字符
	// 如果我们之前遍历了 s1cnt' 个 s1 时，匹配到的是第 s2cnt' 个 s2 中同样的第 index 个字符，那么就有循环节了
	// 我们用 (s1cnt', s2cnt', index) 和 (s1cnt, s2cnt, index) 表示两次包含相同 index 的匹配结果
	// 那么哈希映射中的键就是 index，值就是 (s1cnt', s2cnt') 这个二元组
	// 循环节就是；
	//    - 前 s1cnt' 个 s1 包含了 s2cnt' 个 s2
	//    - 以后的每 (s1cnt - s1cnt') 个 s1 包含了 (s2cnt - s2cnt') 个 s2
	// 那么还会剩下 (n1 - s1cnt') % (s1cnt - s1cnt') 个 s1, 我们对这些与 s2 进行暴力匹配
	// 注意 s2 要从第 index 个字符开始匹配
	unordered_map<int, pair<int, int>> recall;
	pair<int, int> pre_loop, in_loop;
	while (true) {
		// 我们多遍历一个 s1，看看能不能找到循环节
		++s1cnt;
		for (char ch : s1) {
			if (ch == s2[index]) {
				index += 1;
				if (index == s2.size()) {
					++s2cnt;
					index = 0;
				}
			}
		}
		// 还没有找到循环节，所有的 s1 就用完了
		if (s1cnt == n1) {
			return s2cnt / n2;
		}
		// 出现了之前的 index，表示找到了循环节
		if (recall.count(index)) {
			auto s1cnt_prime = recall[index].first, s2cnt_prime = recall[index].second;
			// 前 s1cnt' 个 s1 包含了 s2cnt' 个 s2
			pre_loop = { s1cnt_prime, s2cnt_prime };
			// 以后的每 (s1cnt - s1cnt') 个 s1 包含了 (s2cnt - s2cnt') 个 s2
			in_loop = { s1cnt - s1cnt_prime, s2cnt - s2cnt_prime };
			break;
		}
		else {
			recall[index] = { s1cnt, s2cnt };
		}
	}
	// ans 存储的是 S1 包含的 s2 的数量，考虑的之前的 pre_loop 和 in_loop
	int ans = pre_loop.second + (n1 - pre_loop.first) / in_loop.first * in_loop.second;
	// S1 的末尾还剩下一些 s1，我们暴力进行匹配
	int rest = (n1 - pre_loop.first) % in_loop.first;
	for (int i = 0; i < rest; ++i) {
		for (char ch : s1) {
			if (ch == s2[index]) {
				++index;
				if (index == s2.size()) {
					++ans;
					index = 0;
				}
			}
		}
	}
	// S1 包含 ans 个 s2，那么就包含 ans / n2 个 S2
	return ans / n2;
}


struct listNode
{
	listNode* next, * prev;
};

struct listData
{
	int data;
	listNode node;
};

/*岛屿个数*/
void BFS(vector<vector<char>>& grid, int m, int n, vector<vector<bool>>& isVisited, int x, int y) {
	static int dx[4] = { 0,-1,0,1 }, dy[4] = { -1,0,1,0 };
	struct point {
		int x, y;
		point(int i, int j) { x = i, y = j; }
	};
	queue<point> pqueue;
	pqueue.push(point(x, y));
	while (!pqueue.empty()) {
		point p = pqueue.front();
		pqueue.pop();
		for (int i = 0; i < 4; i++) {
			int nx = p.x + dx[i], ny = p.y + dy[i];
			if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == '1' && !isVisited[nx][ny]) {
				isVisited[nx][ny] = true;
				pqueue.push(point(nx, ny));
			}
		}
	}
}
int numIslands(vector<vector<char>>& grid) {
	if (!grid.size() || !grid[0].size()) {
		return 0;
	}
	int m = grid.size(), n = grid[0].size();
	vector<vector<bool>> isVisited(m, vector<bool>(n, false));
	int ans = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (grid[i][j] == '1' && !isVisited[i][j]) {
				ans++;
				BFS(grid, m, n, isVisited, i, j);
			}
		}
	}
	return ans;
}


int singleNonDuplicate(vector<int>& nums) {
	int left = 0, right = nums.size();
	int k = (right + left) >> 1;
	while (true) {
		if (k == left || k == right - 1) {
			return nums[k];
		}
		if (((right - left) >> 1) & 0x01) {///一半是奇数个
			if (nums[k] == nums[k - 1]) {
				left = k + 1;
			}
			else if (nums[k] == nums[k + 1]) {
				right = k;
			}
			else {
				return nums[k];
			}
		}
		else {
			if (nums[k] == nums[k - 1]) {
				right = k - 1;
			}
			else if (nums[k] == nums[k + 1]) {
				left = k;
			}
			else {
				return nums[k];
			}
		}
		k = (right + left) >> 1;
	}
}

int numberOfSubarrays(vector<int>& nums, int k) {
	vector<pair<int, int>> oddPair;///vector<pair<奇数的下标，该数距离左边第一个相邻奇数的距离>>
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] & 0x01) {
			oddPair.push_back({ i, oddPair.size() > 0 ? i - oddPair[oddPair.size() - 1].first - 1 : i });
		}
	}
	int ans = 0;
	for (int i = 0; i + k - 1 < oddPair.size(); i++) {
		int left = oddPair[i].second;
		int right = i + k >= oddPair.size() ? nums.size() - 1 - oddPair[i + k - 1].first : oddPair[i + k].second;
		ans += (left + 1) * (right + 1);
	}
	return ans;
}



struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};

void rightSideTra(TreeNode* node, vector<int>& nums) {
	if (node) {
		nums.push_back(node->val);
		if (node->right) {
			rightSideTra(node->right, nums);
		}
		else if (node->left) {
			rightSideTra(node->left, nums);
		}
	}
}
/*树的右视图能看到的节点*/
vector<int> rightSideView(TreeNode* root) {
	vector<int> nums;
	queue<pair<TreeNode*, int>> nque;
	int level = -1;
	if (root) nque.push({ root,0 });
	while (!nque.empty()) {
		pair<TreeNode*, int> np = nque.front();
		nque.pop();
		if (np.first->right) {
			nque.push({ np.first->right,np.second + 1 });
		}
		if (np.first->left) {
			nque.push({ np.first->left,np.second + 1 });
		}
		if (level < np.second) {
			nums.push_back(np.first->val);
			level++;
		}
	}
	return nums;
}

bool isPalindrome(string str) {
	int left = 0, right = str.size() - 1;
	while (left < right) {
		if (str[left] != str[right]) {
			return false;
		}
		left++;
		right--;
	}
	return true;
}
/*最近的回文数字*/
string nearestPalindromic(string n) {
	int64_t onum = stoll(n);
	///先求出对称轴
	int left, right, len = n.size();
	if (len & 0x01) {
		left = right = len >> 1;
	}
	else {
		left = (len - 1) >> 1;
		right = len >> 1;
	}
	///只有1位数
	if (left == right && left == 0) {
		return onum - 1 >= 0 ? to_string(onum - 1) : to_string(onum + 1);
	}
	int64_t ans;
	///低位改成和高位一样
	string tmp = n;
	for (int i = right, j = left; i < len; i++, j--) {///i和j一定是对称的
		tmp[i] = tmp[j];
	}
	int64_t num1 = stoll(tmp);
	if (num1 == onum) {
		ans = -1;
	}
	else {
		ans = num1;
	}

	int k = 2;
	while (k--) {//////比较轴中心+1和轴中心-1
		tmp = n;
		int64_t uleft = stoll(tmp.substr(0, left + 1));
		uleft = k ? uleft - 1 : uleft + 1;
		if (uleft == 0) {
			tmp = "9";
		}
		else {
			string t = to_string(uleft);
			tmp[right] = t[t.size() - 1];
			tmp = t + tmp.substr(right == left ? right + 1 : right);
		}
		int i = tmp.length() / 2;
		while (i < tmp.length()) {
			tmp[i] = tmp[tmp.length() - 1 - i];
			i++;
		}
		num1 = stoll(tmp);
		if (ans == -1) {
			ans = num1;
		}
		else {
			ans = abs(num1 - onum) < abs(ans - onum) ? num1 : abs(num1 - onum) == abs(ans - onum) ? min(ans, num1) : ans;
		}
	}

	return to_string(ans);
}

///快排写法一
void quiksort(vector<int>& arr, int begin, int end) {
	if (begin >= end - 1) {
		return;
	}
	int left = begin, right = end - 1;
	int pivot = arr[left];
	while (left < right) {
		while (right > left && arr[right] >= pivot) {
			right--;
		}
		while (left < right && arr[left] < pivot) {
			left++;
		}
		int tmp = arr[left];
		arr[left] = arr[right];
		arr[right] = tmp;
	}
	for (auto n : arr)
		cout << n << ",";
	cout << endl;
	quiksort(arr, left + 1, end);
	quiksort(arr, begin, left + 1);
}
///快排写法二
void Qsort(vector<int>& arr, int low, int high) {
	if (high <= low) return;
	int i = low;
	int j = high + 1;
	int key = arr[low];
	while (true) {
		/*从左向右找比key大的值*/
		while (arr[++i] < key) {
			if (i == high) {
				break;
			}
		}
		/*从右向左找比key小的值*/
		while (arr[--j] > key) {
			if (j == low) {
				break;
			}
		}
		if (i >= j) break;
		/*交换i,j对应的值*/
		int temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}
	/*中枢值与j对应值交换*/
	int temp = arr[low];
	arr[low] = arr[j];
	arr[j] = temp;
	Qsort(arr, low, j - 1);
	Qsort(arr, j + 1, high);
}

/*topK问题*/
void quicksort(vector<int>& arr, int begin, int end, int k) {
	if (begin >= end - 1 || k <= 0) {
		return;
	}
	int left = begin, right = end - 1;
	int pivot = arr[left];
	while (left < right) {
		while (left < right && arr[right] >= pivot) {
			right--;
		}
		while (left < right && arr[left] < pivot) {
			left++;
		}
		int tmp = arr[left];
		arr[left] = arr[right];
		arr[right] = tmp;
	}
	if (left + 1 == k) {
		return;
	}
	else if (left + 1 > k) {
		quicksort(arr, begin, left + 1, k);
	}
	else {
		quicksort(arr, left + 1, end, k);
	}
}

vector<int> getLeastNumbers(vector<int>& arr, int k) {
	quicksort(arr, 0, arr.size(), k);
	vector<int> ans(k, 0);
	copy(arr.begin(), arr.begin() + k, ans.begin());
	stable_sort(ans.begin(), ans.end());
	return ans;
}

///是否有效的二叉搜索树/////
bool isCoreBST(TreeNode* node, TreeNode* maxNode, TreeNode* minNode) {
	if (node == nullptr) {
		return true;
	}
	if (maxNode && node->val >= maxNode->val) {
		return false;
	}
	if (minNode && node->val <= minNode->val) {
		return false;
	}
	return isCoreBST(node->left, node, nullptr) && isCoreBST(node->right, nullptr, node);
}
bool isValidBST(TreeNode* root) {
	return isCoreBST(root, nullptr, nullptr);
}

/*借助栈实现二叉树的中序遍历*/
vector<int> inorderTraversal(TreeNode* root) {
	vector<int> res;
	TreeNode* curr = root;
	stack<TreeNode*> nodeStack;
	while (curr || !nodeStack.empty()) {
		while (curr) {
			nodeStack.push(curr);
			curr = curr->left;
		}
		curr = nodeStack.top();
		nodeStack.pop();
		res.push_back(curr->val);
		curr = curr->right;
	}
	return res;
}

/*递归法求全排列*/
void travPermute(vector<int>& nums, int begin, vector<vector<int>>& res) {
	if (begin == nums.size() - 1) {
		res.push_back(nums);
		return;
	}
	for (int i = begin; i < nums.size(); i++) {
		swap(nums[i], nums[begin]);
		travPermute(nums, begin + 1, res);
		swap(nums[i], nums[begin]);
	}
}
vector<vector<int>> permute(vector<int>& nums) {
	vector<vector<int>> res;
	travPermute(nums, 0, res);
	return res;
}

/*
按照首尾元素有大于1的最大公约数划分子数组，求最小子数组个数，
 O(n^2)，超时
*/
bool hasGCD(int num1, int num2) {
	if (num1 < num2) {
		swap(num1, num2);
	}
	while (int tmp = num1 % num2) {
		num1 = num2;
		num2 = tmp;
	}
	return num2 == 1 ? false : true;
}
int splitArray(vector<int>& nums) {
	int n = nums.size();
	vector<int>f(n, n);
	for (int i = 0; i < n; i++) {
		f[i] = i == 0 ? 1 : f[i - 1] + 1;
		for (int j = 0; j < i; j++) {
			if (hasGCD(nums[i], nums[j])) {
				if (j == 0) {
					f[i] = 1;
					break;
				}
				else {
					f[i] = min(f[i], f[j - 1] + 1);
				}
			}
		}
	}
	return f[n - 1];
}

/*合并有序链表，分治，复杂度O(n*k*logk)，k-链表个数，n-平均每个链表的长度*/
struct ListNode {
	int val;
	ListNode* next;
	ListNode(int x) : val(x), next(NULL) {}
};
ListNode* mergeList(ListNode* list1, ListNode* list2) {
	ListNode res(0), * p = &res;
	while (list1 && list2) {
		if (list1->val <= list2->val) {
			p->next = list1;
			list1 = list1->next;
		}
		else {
			p->next = list2;
			list2 = list2->next;
		}
		p = p->next;
	}
	p->next = list1 ? list1 : list2;
	return res.next;
}
ListNode* mergeKLists(vector<ListNode*>& lists) {
	queue<ListNode*> queList;
	for (int i = 0; i < lists.size(); i++) {
		queList.push(lists[i]);
	}
	while (!queList.empty()) {
		ListNode* lfirst = queList.front();
		ListNode* lsecond;
		queList.pop();
		if (queList.empty()) {
			return lfirst;
		}
		else {
			lsecond = queList.front();
			queList.pop();
		}
		queList.push(mergeList(lfirst, lsecond));
	}
	return nullptr;
}

/*链表排序，O(nlogn)时间复杂度，常数空间复杂度*/
ListNode* sortList(ListNode* head) {
	if (!head || !head->next) {
		return head;
	}
	ListNode* fast = head->next, * slow = head;
	while (fast && fast->next) {
		fast = fast->next->next, slow = slow->next;
	}
	ListNode* mid = slow->next;
	slow->next = nullptr;
	ListNode* left = sortList(head), * right = sortList(mid);
	ListNode res(0), * p = &res;
	while (left && right) {
		if (left->val <= right->val) {
			p->next = left;
			left = left->next;
		}
		else {
			p->next = right;
			right = right->next;
		}
		p = p->next;
	}
	p->next = left ? left : right;
	return res.next;
}

//从前的砝码问题
///给定一组砝码的重量值，每种砝码能用任意个，判定能不能用这些砝码值称出任意数字的重量
bool isAnyWeight(const vector<int>& weights) {
	int len = weights.size();
	if (len < 1 || (len == 1 && weights[0] != 1)) {
		return false;
	}
	///求最大公约数，若最大公约数为1，则符合题意
	int nMin = weights[0], nMax;
	for (int i = 1; i < len; i++) {
		nMax = weights[i];
		if (nMax < nMin) {
			swap(nMax, nMin);
		}
		while (int mod = nMax % nMin) {
			nMax = nMin;
			nMin = mod;
		}
		if (nMin == 1) {
			return true;
		}
	}
	return false;
}

///给定三个字符串，s,t,p，t为目标子串，移动p中的字符插入到s的任意位置（即p中每个位置的字符只能用一次），
///判定能不能由s,p组成和t一模一样的字符串。字符限定为小写的拉丁字母
bool isConstructable(const string& s, const string& t, const string& p) {
	static vector<int> srcStat(26, 0);
	static vector<int> tarStat(26, 0);
	fill(srcStat.begin(), srcStat.end(), 0);
	fill(tarStat.begin(), tarStat.end(), 0);
	int tidx, sidx;
	tidx = sidx = 0;
	int slen = s.length(), tlen = t.length();
	while (sidx < slen && tidx < tlen) {
		char chs = s[sidx], cht = t[tidx];
		tarStat[cht - 'a']++;
		if (chs == cht) {
			srcStat[chs - 'a']++;
			sidx++;
		}
		tidx++;
	}
	if (sidx < slen && tidx == tlen)///s中出现了t中不存在的字符，或者s中字符的顺序和t中的顺序不一致
	{
		return false;
	}
	while (tidx < tlen) {
		tarStat[t[tidx] - 'a']++;
		tidx++;
	}
	for (auto ch : p) {
		srcStat[ch - 'a']++;
	}
	for (int i = 0; i < srcStat.size(); i++) {
		if (srcStat[i] < tarStat[i]) { ///原字符串中的字符数量必须 >= 目标字符串中的字符数量
			return false;
		}
	}
	return true;
}

////搜索旋转排序数组，二分法，O(log n)
int binarySearch(vector<int>& nums, int left, int right, int target) {
	if (left == right) {
		return nums[left] == target ? left : -1;
	}
	int mid = (left + right) >> 1;
	if (nums[mid] == target) {
		return mid;
	}
	else if (nums[mid] > target) {
		if (nums[right] >= nums[mid] && nums[mid] >= nums[left]) {///单调递增区间
			return nums[left] > target ? -1 : binarySearch(nums, left, mid, target);
		}
		else if (nums[right] <= nums[mid] && nums[mid] <= nums[left]) {///单调递减区间
			return nums[right] > target ? -1 : binarySearch(nums, mid + 1, right, target);
		}
		else {///非单调区间
			if (nums[mid] > nums[left]) {
				return nums[left] <= target ? binarySearch(nums, left, mid, target) : binarySearch(nums, mid + 1, right, target);
			}
			else {
				return binarySearch(nums, left, mid, target);
			}
		}
	}
	else {
		if (nums[right] >= nums[mid] && nums[mid] >= nums[left]) {///单调递增区间
			return nums[right] < target ? -1 : binarySearch(nums, mid + 1, right, target);
		}
		else if (nums[right] <= nums[mid] && nums[mid] <= nums[left]) {///单调递减区间
			return nums[left] < target ? -1 : binarySearch(nums, left, mid, target);
		}
		else {///非单调区间
			if (nums[mid] > nums[left]) {
				return binarySearch(nums, mid + 1, right, target);
			}
			else {
				return nums[left] <= target ? binarySearch(nums, left, mid, target) : binarySearch(nums, mid + 1, right, target);
			}
		}
	}
}
int search(vector<int>& nums, int target) {
	return nums.size() ? binarySearch(nums, 0, nums.size() - 1, target) : -1;
}

////一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。
vector<int> singleNumbers(vector<int>& nums) {
	int nxor = 0;
	for (auto n : nums) {
		nxor ^= n;
	}
	///nxor最终结果是两个只出现一次数字的异或
	///nxor某位置的1一定来自于其中一个，取出右边第一个为1的位置
	int n = 0, oldxor = nxor;
	while ((nxor & 0x01) == 0) {
		nxor >>= 1;
		n++;
	}
	int x = 1 << n, y = 0;
	///选取对应位置为1的数字再进行异或，最终得到的数字一定是只出现了一次的那个
	for (auto i : nums) {
		if ((i & x))
			y ^= i;
	}
	return { y, oldxor ^ y };
}

///数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。
///请写一个函数，求任意第n位对应的数字。
int findNthDigit(int n) {
	if (n == 0) {
		return 0;
	}
	n -= 1;
	int nfactor = 9, nbase = 10, npow = 0;
	while (n >= (npow + 1) * nfactor) {
		n -= (npow + 1) * nfactor;
		nfactor *= nbase;
		npow++;
	}
	int nquot = n / (npow + 1), nmod = n % (npow + 1), ntmp = pow(nbase, npow);
	ntmp += nquot;
	string str = to_string(ntmp);
	return str[nmod] - '0';
}

///山脉数组中查找目标值
class MountainArray {
private:
	vector<int> nums;
public:
	int get(int index) { return nums[index]; }
	int length() { return nums.size(); }
	MountainArray(const vector<int>& vec) :nums(vec) {}
};
int binarySearch(MountainArray& mountainArr, int target, int left, int right, int key(int)) {
	target = key(target);
	while (left <= right) {
		int mid = (left + right) >> 1;
		int cur = key(mountainArr.get(mid));
		if (cur == target) {
			return mid;
		}
		else if (cur < target) {
			left = mid + 1;
		}
		else {
			right = mid - 1;
		}
	}
	return -1;
}
int findInMountainArray(int target, MountainArray& mountainArr) {
	int left = 0, right = mountainArr.length() - 1;
	///先找到峰值
	while (left < right) {
		int mid = (left + right) >> 1;
		if (mountainArr.get(mid) < mountainArr.get(mid + 1)) {
			left = mid + 1;
		}
		else {
			right = mid;
		}
	}
	int peak = left;
	int ans = binarySearch(mountainArr, target, 0, peak, [](int x)->int {return x; });
	return ans > -1 ? ans : binarySearch(mountainArr, target, peak, mountainArr.length() - 1, [](int x)->int {return -x; });
}


/// [1, n]的所有数字中，1一共出现的多少次
///思路：分别计算个位/十位/百位...上1出现的次数，个位上的1每隔10出现一次，并且只有1个；十位上的1每隔100出现一次，并且是连续的10个，以此类推。。。
int countDigitOne(int n) {
	int countr = 0;
	for (long long i = 1; i <= n; i *= 10) {
		long long divider = i * 10;
		countr += (n / divider) * i + min(max(n % divider - i + 1, 0LL), i);
	}
	return countr;
}

///无重复字符的最长子串长度
int lengthOfLongestSubstring(string s) {
	///双指针法
	unordered_set<char> chrset;
	int len = s.length();
	///右指针rk
	int rk = -1, ans = 0;
	///遍历左指针
	for (int i = 0; i < len && rk < len; i++) {
		if (i != 0) {
			chrset.erase(s[i - 1]);
		}
		while (rk + 1 < len && !chrset.count(s[rk + 1])) {
			chrset.insert(s[++rk]);
		}
		ans = max(rk - i + 1, ans);
	}
	return ans;
}

///和最大子数组
int maxSubArray(vector<int>& nums) {
	int n = nums.size();
	if (!n) {
		return 0;
	}
	vector<int> nsum(n + 1, 0);
	for (int i = 0; i < n; i++) {
		nsum[i + 1] = nsum[i] + nums[i];
	}
	deque<int> dq;
	int ans = nsum[1];
	dq.push_back(nsum[0]);
	for (int i = 1; i <= n; i++) {
		if (!dq.empty()) {
			ans = max(nsum[i] - dq.front(), ans);
		}
		while (!dq.empty() && nsum[i] < dq.back()) {
			dq.pop_back();
		}
		dq.push_back(nsum[i]);
	}
	return ans;
}

///乘积最大子数组
int maxProduct(vector<int>& nums) {
	////求连续乘积
	//int n = nums.size();
	//vector<long long> products(n);
	//products[0] = nums[0];
	//for (int i = 1; i < n; i++) {
	//	if (products[i - 1] == 0) {
	//		products[i] = nums[i];///碰见为0的点，重新开始计算子数组乘积
	//	}
	//	else {
	//		products[i] = (long long)nums[i] * products[i - 1];///子数组连续乘积
	//	}
	//}
	//long long numNegtive = 0, ans = LLONG_MIN;
	//for (int i = 0; i < n; i++) {
	//	long long tmp = products[i];
	//	if (tmp < 0) {
	//		if (numNegtive != 0) {
	//			tmp /= numNegtive;
	//		}
	//		else {///无效的numNegtive，将其赋值
	//			numNegtive = tmp;
	//		}
	//	}
	//	else if (tmp == 0) {
	//		numNegtive = 0;///使之无效
	//	}
	//	else {
	//		///什么都不做
	//	}
	//	ans = max(ans, tmp);
	//}
	//return ans;

	////动态规划
	int nmax = INT_MIN, imax = 1, imin = 1;
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] < 0) {
			swap(imax, imin);
		}
		imax = max(nums[i] * imax, nums[i]);
		imin = min(nums[i] * imin, nums[i]);
		nmax = max(imax, nmax);
	}
	return nmax;
}

///乘积小于K的子数组（数组中元素都是正整数）
int numSubarrayProductLessThanK(vector<int>& nums, int k) {
	int ans = 0;
	int product = 1, pleft = 0, n = nums.size();
	for (int i = 0; i < n; i++) {
		product *= nums[i];
		while (product >= k && pleft < n) {
			product /= nums[pleft];
			pleft++;
		}
		if (pleft <= i) {
			ans += (i - pleft + 1);
		}
	}
	return ans;
}

///链表相，时间O(n)，空间O(1)
ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
	////用类似于判断链表有环的方法,超时
	//if (!headA || !headB) {
	//	return nullptr;
	//}
	//ListNode* pfast, * pslow, * pALast;
	//pALast = headA;
	//while (pALast) {
	//	if (!pALast->next) {
	//		pALast->next = headB;	///将A链表的尾与B链表的头相连，方便操作
	//		break;
	//	}
	//}
	//pfast = pslow = headA;
	//while (pfast && pslow) {
	//	pslow = pslow->next;
	//	pfast = pfast->next;
	//	if (pfast) {
	//		pfast = pfast->next;
	//	}
	//	if (pfast == pslow) {
	//		break;
	//	}
	//}
	/////此时已经确定一定有相交点
	//if (pfast) {
	//	pfast = headA;
	//	while (pfast && pfast != pslow) {
	//		pfast = pfast->next;
	//		pslow = pslow->next;
	//	}
	//}
	//if (pALast) {
	//	pALast->next = nullptr;
	//}
	//return pfast;

	////双指针法，一个指针a链表走，到底之后转到b，另一个指针从b链表走，到底之后转到a，最终两个指针的相遇点就是相交点
	ListNode* p1 = headA, * p2 = headB;
	while (p1 != p2) {
		if (p1)
			p1 = p1->next;
		else
			p1 = headB;
		if (p2)
			p2 = p2->next;
		else
			p2 = headA;
	}
	return p1;
}

///二叉树的最近公共祖先
bool DFSRoute(TreeNode* p, TreeNode* target, vector<TreeNode*>& route) {
	if (p == target) {
		route.emplace_back(p);
		return true;
	}
	if (p) {
		route.emplace_back(p);
		if (!DFSRoute(p->left, target, route) && !DFSRoute(p->right, target, route)) {
			*(route.rbegin()) = nullptr;
			route.pop_back();
			return false;
		}
		return true;
	}
	return false;
}
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	vector<TreeNode*> pvec, qvec;
	DFSRoute(root, p, pvec);
	DFSRoute(root, q, qvec);
	TreeNode* res = root;
	int r = 0;
	while (r < pvec.size() && r < qvec.size()) {
		if (pvec[r] == qvec[r]) {
			res = pvec[r];
		}
		r++;
	}
	return res;
}

class A
{
public:
	void f1() { cout << "A::f1" << endl; }
	virtual void f3() { cout << "A::f3" << endl; }
	virtual void f2() { cout << "A::f2" << endl; }

};
class B
{
public:
	virtual void f1() { cout << "B::f1" << endl; }
	virtual void f2() { cout << "B::f2" << endl; }
};

////版本号比较，形如 1.0.2 这种格式
bool isLegal(const string& str) {
	if (str.length() == 0) {///空字符
		return false;
	}
	if (!isdigit(str[0])) {///开头字符一定要是数字
		return false;
	}
	if (!isdigit(str[str.length() - 1])) {
		return false;
	}
	bool b = false;
	for (auto c : str) {
		if (isdigit(c)) {
			b = false;
			continue;
		}
		else if (c == '.') {
			if (b)return false;
			b = true;
			continue;
		}
		else
			return false;
	}
	return true;
}

vector<int> dispatchVersion(const string& str) {
	vector<int> version;
	int i = 0, j = 0;
	for (i = 0; i < str.length(); i++) {
		if (str[i] == '.') {
			version.push_back(std::stoi((str.substr(j, i - j))));
			j = i + 1;
		}
	}
	version.push_back(std::stoi(str.substr(j, i - j)));
	return version;
}

int compareVersion(const string& str1, const string& str2) {
	///先判度输入是否合法
	if (!isLegal(str1) || !isLegal(str2)) {
		return -1;
	}
	///分离大小版本号
	vector<int> ver1, ver2;
	ver1 = dispatchVersion(str1);
	ver2 = dispatchVersion(str2);
	while (ver1.size() != ver2.size()) {
		ver1.size() < ver2.size() ? ver1.push_back(0) : ver2.push_back(0);
	}
	///最后做比较，从大版本开始，依次做比较
	int i = 0;
	while (i < ver1.size() && i < ver2.size()) {
		if (ver1[i] > ver2[i]) {
			return 1;
		}
		else if (ver1[i] < ver2[i]) {
			return 2;
		}
		i++;
	}
	return 0;
}

////最低票价
int mincostTickets(vector<int>& days, vector<int>& costs) {
	vector<int> costsTickets(366, 0);
	int t = 0;
	for (auto day : days) {
		while (t < day) {
			costsTickets[t] = costsTickets[max(0, t - 1)];
			t++;
		}
		costsTickets[day] = min(min(costsTickets[max(0, day - 1)] + costs[0], costsTickets[max(0, day - 7)] + costs[1]), costsTickets[max(0, day - 30)] + costs[2]);
		t = day + 1;
	}
	return costsTickets[*days.rbegin()];
}

////另一个树的子树
bool isEqualTree(TreeNode* s, TreeNode* t) {
	return s == t || s && t && s->val == t->val && isEqualTree(s->left, t->left) && isEqualTree(s->right, t->right);
}
bool isSubtree(TreeNode* s, TreeNode* t) {
	if (nullptr == t || s == t) {
		return true;
	}
	queue<TreeNode*> tqueue;
	tqueue.push(s);
	TreeNode* tmp;
	while (!tqueue.empty()) {
		tmp = tqueue.front();
		tqueue.pop();
		if (tmp && tmp->val == t->val && isEqualTree(tmp, t))
			return true;
		if (tmp && tmp->left) {
			tqueue.push(tmp->left);
		}
		if (tmp && tmp->right) {
			tqueue.push(tmp->right);
		}
	}
	return false;
}

///最大正方形，输出其面积
bool matchCondition(vector<vector<char>>& matrix, int left, int right, int top, int bottom) {
	///判定是否符合条件,水平方向，坐标范围(bottom,right->left)；竖直方向，坐标范围(bottom->top,right)
	int idx = 0;
	///正方形，宽高相等，可以放在一起判断
	while (right - idx >= left && '1' == matrix[bottom][right - idx] && bottom - idx >= top && '1' == matrix[bottom - idx][right]) {
		idx++;
	}
	if (right - idx >= left) {
		return false;
	}
	return true;
}
int maximalSquare(vector<vector<char>>& matrix) {
	///暴力法
	//if (matrix.size() == 0 || matrix[0].size() == 0) {
	//	return 0;
	//}
	//int m = matrix.size(), n = matrix[0].size();
	//int maxWidth = 0;
	//for (int i = 0; i < m; i++) {
	//	for (int j = 0; j < n; j++) {
	//		if (matrix[i][j] == '1') {
	//			int maxSqr = 1;
	//			while (j + maxSqr < n && i + maxSqr < m && matchCondition(matrix, j, j + maxSqr, i, i + maxSqr)) {
	//				maxSqr++;
	//			}
	//			maxWidth = max(maxWidth, maxSqr);
	//		}
	//	}
	//}
	//return maxWidth * maxWidth;

	/////动态规划，dp[i][j]表示以点（i，j）为右下角的最大正方形边长，则dp(i,j) = min(dp(i−1,j),dp(i−1,j−1),dp(i,j−1))+1
	if (matrix.size() == 0 || matrix[0].size() == 0) {
		return 0;
	}
	int maxSide = 0;
	int rows = matrix.size(), columns = matrix[0].size();
	vector<vector<int>> dp(rows, vector<int>(columns));///只用到了左方，左上方，上方的值，可优化为一维数组
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			if (matrix[i][j] == '1') {
				if (i == 0 || j == 0) {
					dp[i][j] = 1;
				}
				else {
					dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
				}
				maxSide = max(maxSide, dp[i][j]);
			}
		}
	}
	return maxSide * maxSide;
}

///实现 pow(x, n) ，即计算 x 的 n 次幂函数。
double myPow(double x, int n) {
	//递归实现
	if (n == 0) {
		return 1;
	}
	double d = myPow(x, n / 2);
	return (n & 1 ? (n < 0 ? 1.0 / x : x) : 1) * d * d;

	//迭代法实现
	//long long N = n;
	//return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);
}
double quickMul(double x, long long N) {
	double ans = 1.0;
	// 贡献的初始值为 x
	double x_contribute = x;
	// 在对 N 进行二进制拆分的同时计算答案
	while (N > 0) {
		if (N % 2 == 1) {
			// 如果 N 二进制表示的最低位为 1，那么需要计入贡献
			ans *= x_contribute;
		}
		// 将贡献不断地平方
		x_contribute *= x_contribute;
		// 舍弃 N 二进制表示的最低位，这样我们每次只要判断最低位即可
		N /= 2;
	}
	return ans;
}

///你的任务是计算 a^b 对 1337 取模，a 是一个正整数，b 是一个非常大的正整数且会以数组形式给出。
int quickMulMod(int a, int b) {
	int ans = 1;
	int cbute = a;
	while (b > 0) {
		if (b % 2 == 1) {
			ans *= cbute;
			ans %= 1337;
		}
		cbute *= cbute;
		cbute %= 1337;
		b /= 2;
	}
	return ans;
}
int superPow(int a, vector<int>& b) {
	a %= 1337;
	int ans = 1;
	for (int i = 0; i < b.size(); i++) {
		ans = quickMulMod(ans, 10);
		int t = quickMulMod(a, b[i]);
		ans *= t;
		ans %= 1337;
	}
	return ans;
}

///逃离大迷宫
//在一个 10 ^ 6 x 10 ^ 6 的网格中，每个网格块的坐标为 (x, y)，其中 0 <= x, y < 10 ^ 6。
//我们从源方格 source 开始出发，意图赶往目标方格 target。每次移动，我们都可以走到网格中在四个方向上相邻的方格，只要该方格不在给出的封锁列表 blocked 上。
//只有在可以通过一系列的移动到达目标方格时才返回 true。否则，返回 false。
static int widthLimit = 1000000;
bool bfsTrace(const vector<int>& source, const vector<int>& target, unordered_set<string>& blocks) {
	static int delta[4][2] = { {0,-1},{-1,0} ,{0,1},{1,0} };
	unordered_set<string> visited;
	visited.insert(to_string(source[0]) + ":" + to_string(source[1]));
	queue<pair<int, int>> que;
	que.push({ source[0],source[1] });
	pair<int, int> p;
	string str;
	int maxCount = blocks.size() * (blocks.size() - 1) / 2;
	while (!que.empty()) {
		p = que.front();
		que.pop();
		for (int i = 0; i < 4; i++) {
			int x = p.first + delta[i][0];
			int y = p.second + delta[i][1];
			if (x >= 0 && x < widthLimit && y >= 0 && y < widthLimit) {
				str = to_string(x) + ":" + to_string(y);
				if (!blocks.count(str) && !visited.count(str)) {
					if (x == target[0] && y == target[1]) {
						return true;
					}
					visited.insert(str);
					que.push({ x,y });
				}
			}
		}
		// blocked 的 大小为 length
		// 如果使用这 length 个 block 可以围成最大的区域是 length*(length-1)/2，如下：
		/*
			0th      _________________________
					|O O O O O O O X
					|O O O O O O X
					|O O O O O X
					|O O O O X
					.O O O X
					.O O X
					.O X
			length  |X
		从上面可以计算出 block（即 X）可以围城的最大区域(是一个角的三角形)，大小计算如下：
		1 + 2 + 3 + 4 + ... + (length-1) = length*(length-1)/2
		*/
		// 也就是说，如果迭代了 length*(length-1)/2 步还能继续走的话，那么是肯定可以到达 target 的
		if (visited.size() > maxCount) {
			return true;
		}
	}
	return false;
}
bool isEscapePossible(vector<vector<int>>& blocked, vector<int>& source, vector<int>& target) {
	unordered_set<string> blocks;
	for (auto& v : blocked) {
		blocks.insert(to_string(v[0]) + ":" + to_string(v[1]));
	}
	return bfsTrace(source, target, blocks) && bfsTrace(target, source, blocks);
}

///双栈结构实现最小栈getMin in O(1) time
class MinStack {
	stack<int> x_stack;
	stack<int> min_stack;
public:
	MinStack() {
		min_stack.push(INT_MAX);
	}

	void push(int x) {
		x_stack.push(x);
		min_stack.push(min(min_stack.top(), x));
	}

	void pop() {
		x_stack.pop();
		min_stack.pop();
	}

	int top() {
		return x_stack.top();
	}

	int getMin() {
		return min_stack.top();
	}
};

///二叉树的层序遍历
void dfsLevelOrder(TreeNode* node, int curl, vector<vector<int>>& res) {
	if (curl == res.size()) {
		res.push_back(vector<int>());
	}
	res[curl].push_back(node->val);
	if (node->left) {
		dfsLevelOrder(node->left, curl + 1, res);
	}
	if (node->right) {
		dfsLevelOrder(node->right, curl + 1, res);
	}
}
vector<vector<int>> levelOrder(TreeNode* root) {
	vector<vector<int>> ans;
	if (root) {
		dfsLevelOrder(root, 0, ans);
	}
	return ans;
}

/////和为K的子数组
int subarraySum(vector<int>& nums, int k) {
	///可以省略求和数组
	//vector<int> sumVec(nums.size());
	//multimap<int, int> sumMap;
	//int ans = 0;
	//sumMap.insert({ 0, -1 });
	//for (int i = 0; i < nums.size(); i++) {
	//	sumVec[i] = i > 0 ? sumVec[i - 1] + nums[i] : nums[i];
	//	ans += sumMap.count(sumVec[i] - k);
	//	sumMap.insert({ sumVec[i], i });
	//}
	//return ans;
	int numSum = 0;
	multimap<int, int> sumMap;
	int ans = 0;
	sumMap.insert({ 0, -1 });
	for (int i = 0; i < nums.size(); i++) {
		numSum = i > 0 ? numSum + nums[i] : nums[i];
		ans += sumMap.count(numSum - k);
		sumMap.insert({ numSum, i });
	}
	return ans;
}

////K 个一组翻转链表
ListNode* reverseKGroup(ListNode* head, int k) {
	int n = 0;
	ListNode* nextTail = nullptr, * preTail = nullptr;
	ListNode* ppre, * pcur, * pnext;
	ListNode* ans = nullptr;
	ppre = nullptr, pcur = head;
	while (pcur) {
		if (n == k - 1) {
			ans = pcur;
		}
		if (n % k == 0) {
			nextTail = pcur;
		}
		pnext = pcur->next;
		pcur->next = ppre;
		ppre = pcur;
		pcur = pnext;
		if (n % k == k - 1) {
			if (preTail) {
				preTail->next = ppre;
			}
			preTail = nextTail;
		}
		n++;
	}
	if (nextTail)///尾指针设置
		nextTail->next = nullptr;

	n--;///n回踩
	///重新翻转最后一组少于k个的节点
	if (n % k != k - 1) {
		pcur = ppre;
		ppre = nullptr;
		while (pcur) {
			pnext = pcur->next;
			pcur->next = ppre;
			ppre = pcur;
			pcur = pnext;
		}
		if (preTail) {
			preTail->next = ppre;
		}
	}

	return ans ? ans : head;
}

/// 课程表 II，考察拓扑排序
vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
	vector<int> indegree(numCourses, 0);
	vector<vector<int>> edges(numCourses);
	for (auto& it : prerequisites) {
		indegree[it[0]]++;
		edges[it[1]].push_back(it[0]);
	}
	vector<int> ans;
	int reqHead = -1, nextHead = -1;
	for (int i = 0; i < numCourses; i++) {
		nextHead = -1;
		if (reqHead < 0) {
			for (int j = 0; j < numCourses; j++) {
				if (indegree[j] == 0) {
					reqHead = j;
					break;
				}
			}
		}
		if (reqHead > -1) {
			ans.emplace_back(reqHead);
			for (int j = 0; j < edges[reqHead].size(); j++) {
				--indegree[edges[reqHead][j]] == 0 ? nextHead = edges[reqHead][j] : nextHead = nextHead;
			}
			indegree[reqHead]--;
			reqHead = nextHead;
		}
		else {///提前退出
			break;
		}
	}
	return ans.size() == numCourses ? ans : vector<int>{};
}

///除自身以外数组的乘积
vector<int> productExceptSelf(vector<int>& nums) {
	///ans[i] = (nums[0]*nums[1]...nums[i-1]) * (nums[i+1]*nums[i+2]....nums[n-1])
	///取前缀与后缀的乘积相乘即可，未节省空间，可将求解过程分为两步，1--求前缀乘积，2--求后缀乘积，并将1中得到的前缀与后缀相乘
	vector<int> ans(nums.size(), 1);
	int temp = 1;
	for (int i = 0; i < nums.size() - 1; i++) {
		temp *= nums[i];
		ans[i + 1] = temp;
	}
	temp = 1;
	for (int i = nums.size() - 1; i > 0; i--) {
		temp *= nums[i];
		ans[i - 1] *= temp;
	}
	return ans;
}

///接雨水
int trap(vector<int>& height) {
	deque<pair<int, int>> pdeuqe;
	int ans = 0, btm = 0;
	for (int i = 0; i < height.size(); i++) {
		int heit = height[i];
		while (!pdeuqe.empty() && heit >= pdeuqe.back().first) {
			pair<int, int>tpair = pdeuqe.back();
			ans += max(tpair.first - btm, 0) * max(i - tpair.second - 1, 0);
			btm = tpair.first;
			pdeuqe.pop_back();
		}
		if (pdeuqe.empty()) {
			btm = 0;
		}
		else {
			ans += (heit - btm) * max(i - pdeuqe.back().second - 1, 0);
			btm = heit;
		}
		pdeuqe.push_back({ heit,i });
	}
	return ans;
}

/// 回文链表
ListNode* reverseList(ListNode* head) {
	ListNode* pre = nullptr, * pcur = head, * pnext;
	while (pcur) {
		pnext = pcur->next;
		pcur->next = pre;
		pre = pcur;
		pcur = pnext;
	}
	return pre;
}
bool isPalindrome(ListNode* head) {
	if (!head || !head->next) {
		return true;
	}
	ListNode* ptwice = head, * ponce = head;
	while (ptwice && ptwice->next) {
		ptwice = ptwice->next->next;
		if (ptwice) ponce = ponce->next;
	}
	ListNode* pright = ponce->next;
	ponce->next = nullptr;
	ListNode* pleft = reverseList(head);

	bool ans = true;
	if (ptwice) {///奇数个节点
		ponce = pleft->next;
	}
	else if (pleft->val == pleft->next->val) {
		ponce = pleft->next->next;
	}
	else {
		ans = false;
	}
	if (ans) {
		ptwice = pright;
		while (ponce) {
			if (ponce->val != ptwice->val) {
				reverseList(pleft);
				pleft->next = pright;
				return false;
			}
			ponce = ponce->next, ptwice = ptwice->next;
		}
	}
	reverseList(pleft);
	pleft->next = pright;
	return ans;
}

//// 每个元音包含偶数次的最长子字符串
int findTheLongestSubstring(string s) {
	int ans = 0, status = 0, n = s.length();
	vector<int> pos(1 << 5, -1);
	pos[0] = 0;
	///二进制压缩编码，每个位代表对应元音字母出现的奇偶性
	for (int i = 0; i < n; ++i) {
		switch (s[i]) {
		case 'a':
			status ^= 1 << 0;
			break;
		case 'e':
			status ^= 1 << 1;
			break;
		case 'i':
			status ^= 1 << 2;
			break;
		case 'o':
			status ^= 1 << 3;
			break;
		case 'u':
			status ^= 1 << 4;
			break;
		default:
			break;
		}
		if (~pos[status]) {///pos[status]!=-1
			ans = max(ans, i + 1 - pos[status]);
		}
		else {///pos[status]==-1
			pos[status] = i + 1;
		}
	}
	return ans;
}

///从前序与中序遍历序列构造二叉树
TreeNode* buildTree(vector<int>& preorder, int preLeft, int preRight, vector<int>& inorder, int inLeft, int inRight) {
	if (inLeft > inRight) {
		return nullptr;
	}
	TreeNode* node = new TreeNode(preorder[preLeft]);
	int n = inLeft;
	while (n <= inRight) {
		if (inorder[n] == preorder[preLeft]) {
			break;
		}
		n++;
	}
	int m = preLeft + n - inLeft;

	node->left = buildTree(preorder, preLeft + 1, m, inorder, inLeft, n - 1);
	node->right = buildTree(preorder, m + 1, preRight, inorder, n + 1, inRight);
	return node;
}
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
	return buildTree(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
}

///最小覆盖子串(给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字符的最小子串)
///滑动窗口法，窗口右指针尽可能满足条件，左指针尽可能使窗口长度小
string minWindow(string s, string t) {
	unordered_map<char, int> tmap, smap;
	for (auto c : t) {
		tmap[c]++;
	}
	int total = 0;
	int i = 0, minLen = s.size(), minPos = 0;
	for (int j = 0; j < s.length(); j++) {
		if (tmap.count(s[j])) {
			if (smap[s[j]] < tmap[s[j]]) {
				total++;
			}
			smap[s[j]]++;
			if (total == t.size()) {
				int k = i;
				while (k < j) {
					if (tmap.count(s[k])) {
						if (smap[s[k]] - 1 >= tmap[s[k]]) {
							smap[s[k]]--;
						}
						else {
							break;
						}
					}
					k++;
				}
				i = k;
				if (j - i + 1 < minLen) {
					minLen = j - i + 1;
					minPos = i;
				}
			}
		}
	}
	return total < t.size() ? string() : s.substr(minPos, minLen);
}

///寻找两个正序数组的中位数,要求算法的时间复杂度为 O(log(m + n))
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
	if (nums1.size() > nums2.size()) {
		return findMedianSortedArrays(nums2, nums1);
	}

	int m = nums1.size();
	int n = nums2.size();
	int left = 0, right = m, ansi = -1;
	// median1：前一部分的最大值
	// median2：后一部分的最小值
	int median1 = 0, median2 = 0;

	while (left <= right) {
		// 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]
		// 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
		int i = (left + right) / 2;
		int j = (m + n + 1) / 2 - i;

		// nums_im1, nums_i, nums_jm1, nums_j 分别表示 nums1[i-1], nums1[i], nums2[j-1], nums2[j]
		int nums_im1 = (i == 0 ? INT_MIN : nums1[i - 1]);
		int nums_i = (i == m ? INT_MAX : nums1[i]);
		int nums_jm1 = (j == 0 ? INT_MIN : nums2[j - 1]);
		int nums_j = (j == n ? INT_MAX : nums2[j]);

		if (nums_im1 <= nums_j) {
			ansi = i;
			median1 = max(nums_im1, nums_jm1);
			median2 = min(nums_i, nums_j);
			left = i + 1;
		}
		else {
			right = i - 1;
		}
	}

	return (m + n) % 2 == 0 ? (median1 + median2) / 2.0 : median1;
}

/// 和可被 K 整除的子数组
int subarraysDivByK(vector<int>& A, int K) {
	unordered_map<int, int> record{ { 0,1 } };
	int sum = 0, ans = 0;
	for (auto n : A) {
		sum += n;
		int mod = (sum % K + K) % K;
		if (record.count(mod)) {
			ans += record[mod];
		}
		record[mod]++;
	}
	return ans;
}

///字符串解码
string decodeString(string s) {
	stack<string> strstack;
	int i = 0, j = 0;
	for (; i < s.length(); i++) {
		if (s[i] == '[') {
			strstack.push("[");
		}
		else if (s[i] == ']') {
			string strtmp = strstack.top();
			string strpush;

			if (strtmp == "[") {
				strtmp = "";
				strstack.pop();
			}
			else {
				strstack.pop();///
				strstack.pop();///pop [
			}
			if (!strstack.empty() && strstack.top() != "[") {
				string numtmp = strstack.top();
				strstack.pop();
				if (isdigit(numtmp[0])) {
					int num = stoi(numtmp);
					while (num-- > 0) {
						strpush += strtmp;
					}
				}
				else {
					strpush = numtmp + strtmp;
				}
				while (!strstack.empty() && strstack.top() != "[") {
					strpush = strstack.top() + strpush;
					strstack.pop();
				}
			}
			else {
				strpush = strtmp;
			}
			strstack.push(strpush);
		}
		else if (isdigit(s[i])) {
			j = i + 1;
			while (j < s.length() && isdigit(s[j])) {
				j++;
			}
			strstack.push(s.substr(i, j - i));
			i = j - 1;
		}
		else {
			j = i + 1;
			while (j < s.length() && isalpha(s[j])) {
				j++;
			}
			string sss = s.substr(i, j - i);
			while (!strstack.empty() && strstack.top() != "[") {
				sss = strstack.top() + sss;
				strstack.pop();
			}
			strstack.push(sss);
			i = j - 1;
		}
	}
	string ans;
	while (!strstack.empty()) {
		ans = strstack.top() + ans;
		strstack.pop();
	}
	return ans;
}

///198. 打家劫舍
int rob(vector<int>& nums) {
	int n = nums.size();
	int f1, f2, ans = 0;
	f1 = n > 0 ? nums[0] : 0;
	f2 = n > 1 ? max(nums[0], nums[1]) : f1;
	for (int i = 2; i < n; i++) {
		ans = max(f1 + nums[i], f2);
		f1 = f2;
		f2 = ans;
	}
	ans = f2;
	return ans;
}

///84. 柱状图中最大的矩形
int largestRectangleArea(vector<int>& heights) {
	//deque<pair<int, int>> pdeuqe;///pair<height,position>
	//int ans = 0, stpos = -1, fn = 0;
	//for (int i = 0; i < heights.size(); i++) {
	//	int tmp = 0, heit = heights[i];
	//	////高度为0
	//	if (heit == 0) {
	//		pdeuqe.clear();
	//		stpos = i;
	//		continue;
	//	}
	//	////
	//	fn = heit;
	//	int lastpos = stpos;
	//	auto it = pdeuqe.begin();
	//	for (; it != pdeuqe.end(); it++) {
	//		tmp = min(it->first, heit) * (i - lastpos);
	//		fn = max(tmp, fn);
	//		lastpos = it->second;
	//	}

	//	ans = max(fn, ans);
	//	while (!pdeuqe.empty() && heit <= pdeuqe.back().first) {
	//		pdeuqe.pop_back();
	//	}
	//	pdeuqe.push_back({ heit, i });
	//}
	//return ans;

	///以某个位置作为高的矩形最大面积，只要在两边分别找出比该位置高度小的第一个柱子就行
	///以单调栈来实现（递增），
	int ans = 0;
	stack<int> monostk;
	///两边增加0，方便边界处理
	vector<int> new_height(heights.size() + 2, 0);
	for (size_t i = 0; i < heights.size(); i++) {
		new_height[i + 1] = heights[i];
	}
	for (size_t i = 0; i < new_height.size(); i++) {
		while (!monostk.empty() && new_height[i] < new_height[monostk.top()]) {
			int cur = monostk.top();
			monostk.pop();
			///i为cur位置右边第一个比cur高度小的点，monostk栈顶为左边第一个高度比cur位置小的点
			ans = max(ans, new_height[cur] * ((int)i - monostk.top() - 1));
		}
		monostk.push(i);
	}
	return ans;
}

///101. 对称二叉树
bool isSymmetric(TreeNode* left, TreeNode* right) {
	if (left == nullptr && right == nullptr) {
		return true;
	}
	if (left && right) {
		return left->val == right->val && isSymmetric(left->left, right->right) && isSymmetric(left->right, right->left);
	}
	return false;
}
bool isSymmetric(TreeNode* root) {
	if (root == nullptr) return true;
	TreeNode* left = root->left, * right = root->right;
	return isSymmetric(left, right);
}

///837. 新21点
double new21Game(int N, int K, int W) {
	if (K == 0) {
		return 1;
	}
	vector<double> dp(K + W, 0);
	for (int i = K; i < K + W && i <= N; i++) {
		dp[i] = 1;
	}
	dp[K - 1] = (double)(min(N - K + 1, W)) / W;
	for (int i = K - 2; i >= 0; i--) {
		dp[i] = dp[i + 1] + (dp[i + 1] - dp[i + W + 1]) / W;
	}
	return dp[0];
}

//128. 最长连续序列
int longestConsecutive(vector<int>& nums) {
	std::unordered_map<int, int> numMap;
	for (size_t i = 0; i < nums.size(); i++) {
		numMap[nums[i]]++;
	}
	int ans = 0;
	int left, right, tmp;
	for (size_t i = 0; i < nums.size(); i++) {
		tmp = numMap.count(nums[i]);///重复的数字只能算一个，所以只需要hashset就可以
		if (tmp == 0) {
			continue;
		}
		left = nums[i] - 1;
		right = nums[i] + 1;
		int t = 0;
		while (t = numMap.count(left)) {
			tmp += t;
			numMap.erase(left);
			left -= 1;
		}
		while (t = numMap.count(right)) {
			tmp += t;
			numMap.erase(right);
			right += 1;
		}
		ans = max(ans, tmp);
	}
	return ans;
}

///990. 等式方程的可满足性
///并查集
class UnionFind {
private:
	vector<int> parent;

public:
	UnionFind() {
		parent.resize(26);
		iota(parent.begin(), parent.end(), 0);
	}

	int find(int index) {
		if (index == parent[index]) {
			return index;
		}
		parent[index] = find(parent[index]);
		return parent[index];
	}

	void unite(int index1, int index2) {
		parent[find(index1)] = find(index2);
	}
};
bool equationsPossible(vector<string>& equations) {
	UnionFind uf;
	for (const string& str : equations) {
		if (str[1] == '=') {
			int idx1 = str[0] - 'a';
			int idx2 = str[3] - 'a';
			uf.unite(idx1, idx2);
		}
	}
	for (const string& str : equations) {
		if (str[1] == '!') {
			int idx1 = str[0] - 'a';
			int idx2 = str[3] - 'a';
			if (uf.find(idx1) == uf.find(idx2)) {
				return false;
			}
		}
	}
	return true;
}

///面试题46. 把数字翻译成字符串
int translateNum(int num) {
	string	numstr = to_string(num);
	int f1 = 0, f2 = 0, f3 = 0;
	f1 = 1;
	f2 = numstr.length() > 1 && stoi(numstr.substr(0, 2)) < 26 ? 2 : 1;
	for (size_t i = 2; i < numstr.length(); i++) {
		f3 = numstr[i - 1] != '0' && stoi(numstr.substr(i - 1, 2)) < 26 ? f1 + f2 : f2;
		f1 = f2;
		f2 = f3;
	}
	return f2;
}

///面试题53 - I. 在排序数组中查找数字 I
int shunxuSearch(vector<int>& nums, int target) {
	int left = 0, right = nums.size();
	int pos = -1;
	while (left < right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] == target) {
			pos = mid;
			break;
		}
		else if (nums[mid] < target) {
			left = mid + 1;
		}
		else {
			right = mid;
		}
	}
	if (pos != -1) {
		int ans = 1, t = pos;
		while (--t >= 0 && nums[t] == target) {
			ans++;
		}
		t = pos;
		while (++t < nums.size() && nums[t] == target) {
			ans++;
		}
		return ans;
	}
	return 0;
}

///1238. 循环码排列
///Gray code
vector<int> circularPermutation(int n, int start) {
	vector<int> ans;
	for (int i = 0; i < pow(2, n); i++) {
		ans.push_back(i ^ (i >> 1));
	}
	size_t i = 0;
	for (; i < ans.size(); i++) {
		if (ans[i] == start) {
			break;
		}
	}
	rotate(ans.begin(), ans.begin() + i, ans.end());
	return ans;
}

///739. 每日温度
vector<int> dailyTemperatures(vector<int>& T) {
	///维护递减队列，递减队列中的数字暂时无法确定第一个比他值大的位置
	deque<int> decendIdxQue;
	vector<int> ans(T.size(), 0);
	for (size_t i = 0; i < T.size(); i++) {
		int t = T[i], p;
		while (!decendIdxQue.empty() && T[p = decendIdxQue.back()] < t) {
			ans[p] = i - p;
			decendIdxQue.pop_back();
		}
		decendIdxQue.push_back(i);
	}
	return ans;
}

///496. 下一个更大元素 I
vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
	unordered_map<int, int> numMap;///<val, position>
	for (size_t i = 0; i < nums2.size(); i++) {
		numMap[nums2[i]] = i;
	}
	vector<int> ans;
	for (size_t i = 0; i < nums1.size(); i++) {
		auto it = numMap.find(nums1[i]);
		if (it != numMap.end()) {
			int t = it->second;
			while (t < nums2.size() && nums2[t] <= nums1[i]) {
				t++;
			}
			ans.push_back(t == nums2.size() ? -1 : nums2[t]);
		}
		else {
			ans.push_back(-1);
		}
	}
	return ans;
}
///15. 三数之和
vector<vector<int>> threeSum(vector<int>& nums) {
	int n = nums.size();
	sort(nums.begin(), nums.end());
	vector<vector<int>> ans;
	// 枚举 a
	for (int first = 0; first < n; ++first) {
		// 需要和上一次枚举的数不相同
		if (first > 0 && nums[first] == nums[first - 1]) {
			continue;
		}
		// c 对应的指针初始指向数组的最右端
		int third = n - 1;
		int target = -nums[first];
		// 枚举 b
		for (int second = first + 1; second < n; ++second) {
			// 需要和上一次枚举的数不相同
			if (second > first + 1 && nums[second] == nums[second - 1]) {
				continue;
			}
			// 需要保证 b 的指针在 c 的指针的左侧
			while (second < third && nums[second] + nums[third] > target) {
				--third;
			}
			// 如果指针重合，随着 b 后续的增加
			// 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
			if (second == third) {
				break;
			}
			if (nums[second] + nums[third] == target) {
				ans.push_back({ nums[first], nums[second], nums[third] });
			}
		}
	}
	return ans;
}

///16. 最接近的三数之和
//排序+双指针, abs(a+b+c - target)
int threeSumClosest(vector<int>& nums, int target) {
	int n = nums.size();
	sort(nums.begin(), nums.end());
	int ans = 1e7;//最优解
	//枚举a
	for (int i = 0; i < n - 2; i++) {
		int a = nums[i];
		// 保证和上一次枚举的元素不相等
		if (i > 0 && nums[i] == nums[i - 1]) {
			continue;
		}
		//使用双指针枚举b和c
		int pb = i + 1, pc = n - 1;
		while (pb < pc) {
			int sum = nums[i] + nums[pb] + nums[pc];
			// 如果和为 target 直接返回答案
			if (sum == target) {
				return sum;
			}
			//更新最优解
			if (abs(sum - target) < abs(ans - target)) {
				ans = sum;
			}
			if (sum > target) {
				// 如果和大于 target，移动 c 对应的指针
				int ppc = pc - 1;
				while (pb < ppc && nums[ppc] == nums[pc]) {
					--ppc;
				}
				pc = ppc;
			}
			else {
				// 如果和小于 target，移动 b 对应的指针
				int ppb = pb + 1;
				while (ppb < pc && nums[ppb] == nums[pb]) {
					ppb++;
				}
				pb = ppb;
			}
		}
	}
	return ans;
}

///1300. 转变数组后最接近目标值的数组和
int findBestValue(vector<int>& arr, int target) {
	int meanVal = 0;
	vector<int> upMeansPos(arr.size());
	iota(upMeansPos.begin(), upMeansPos.end(), 0);
	while (upMeansPos.size()) {
		if (upMeansPos.size() == 1) {
			meanVal = min(target, arr[upMeansPos[0]]);
			break;
		}
		double d = (double)target / upMeansPos.size();
		if (d - (int)d > 0.5) {
			meanVal = (int)d + 1;
		}
		else {
			meanVal = (int)d;
		}
		vector<int> tmpvec;
		for (size_t i = 0; i < upMeansPos.size(); i++) {
			int pos = upMeansPos[i];
			if (arr[pos] <= meanVal) {
				target -= arr[pos];
			}
			else {
				tmpvec.push_back(pos);
			}
		}
		if (upMeansPos.size() == tmpvec.size()) {
			break;
		}
		else {
			upMeansPos.swap(tmpvec);
			tmpvec.clear();
		}
	}

	return meanVal;
}

///123. 买卖股票的最佳时机 III
int maxProfit(vector<int>& prices) {
	int dp_i10 = 0, dp_i11 = INT_MIN;
	int dp_i20 = 0, dp_i21 = INT_MIN;
	for (int price : prices) {
		dp_i20 = max(dp_i20, dp_i21 + price);
		dp_i21 = max(dp_i21, dp_i10 - price);
		dp_i10 = max(dp_i10, dp_i11 + price);
		dp_i11 = max(dp_i11, -price);
	}
	return dp_i20;
}

///最长公共前缀
string longestCommonPrefix(vector<string>& strs) {
	string ans;
	if (strs.size() > 0) {
		ans = strs[0];
		for (size_t i = 1; i < strs.size(); i++) {
			int j = 0;
			while (j < ans.length() && j < strs[i].length() && ans[j] == strs[i][j]) {
				j++;
			}
			ans = ans.substr(0, j);
		}
	}
	return ans;
}

///780. 到达终点
bool reachingPoints(int sx, int sy, int tx, int ty) {
	///注意条件， sx, sy, tx, ty 是范围在 [1, 10^9] 的整数。
	//while (tx >= sx && ty >= sy) {
	//	if (tx == sx && ty == sy) {
	//		return true;
	//	}
	//	if (tx > ty) {
	//		tx -= ty;
	//	}
	//	else {
	//		ty -= tx;
	//	}
	//}
	//return false;

	///用取模加速
	while (tx >= sx && ty >= sy) {
		if (tx == ty) break;
		if (tx > ty) {
			if (ty > sy) tx %= ty;
			else return (tx - sx) % ty == 0;
		}
		else {
			if (tx > sx) ty %= tx;
			else return (ty - sy) % tx == 0;

		}
	}
	return (tx == sx && ty == sy);
}

///297. 二叉树的序列化与反序列化
class Codec {
public:

	// Encodes a tree to a single string.
	string serialize(TreeNode* root) {
		deque<TreeNode*> dqnode;
		string ansstr = "[";
		dqnode.push_back(root);
		TreeNode* node;
		while (!dqnode.empty()) {
			node = dqnode.front();
			dqnode.pop_front();
			if (node != nullptr) {
				ansstr += (to_string(node->val) + ",");
				dqnode.push_back(node->left);
				dqnode.push_back(node->right);
			}
			else {
				ansstr += "null,";
			}
		}
		if (ansstr[ansstr.length() - 1] == ',') {
			ansstr[ansstr.length() - 1] = ']';
		}
		else {
			ansstr.push_back(']');
		}
		return ansstr;
	}

	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data) {
		data = data.substr(1, data.length() - 2);
		size_t pos = 0, t = 0;
		vector<string> strVec;
		while (t != std::string::npos) {
			t = data.find(',', pos);
			strVec.push_back(data.substr(pos, t - pos));
			pos = t + 1;
		}
		deque<TreeNode*> dqnode;
		TreeNode* node, * ans = nullptr;
		if (strVec.size() > 0 && strVec[0] != "null") {
			ans = new TreeNode(stoi(strVec[0]));
			dqnode.push_back(ans);
		}
		for (size_t i = 1; i < strVec.size();) {
			if (!dqnode.empty()) {
				node = dqnode.front();
				dqnode.pop_front();
				if (strVec[i] != "null") {
					dqnode.push_back(new TreeNode(stoi(strVec[i])));
					node->left = dqnode.back();
				}
				i++;
				if (i < strVec.size() && strVec[i] != "null") {
					dqnode.push_back(new TreeNode(stoi(strVec[i])));
					node->right = dqnode.back();
				}
				i++;
			}
		}
		return ans;
	}
};

///1014. 最佳观光组合
int maxScoreSightseeingPair(vector<int>& A) {
	int ans = 0, fn = 0;
	for (size_t i = 1; i < A.size(); i++) {
		fn = max(A[i] - A[i - 1] - 1 + fn, A[i] + A[i - 1] - 1);
		ans = max(fn, ans);
	}
	return ans;
}

///1028. 从先序遍历还原二叉树
TreeNode* recoverFromPreorder(string S) {
	if (S.size() == 0) {
		return nullptr;
	}
	int t = 0;
	while (t < S.size() && S[t] != '-') {
		t++;
	}
	TreeNode* ans = new TreeNode(stoi(S.substr(0, t)));
	TreeNode* tmpNode, * childNode;
	stack<TreeNode*> nodeStack;
	nodeStack.push(ans);
	for (size_t i = t; i < S.length(); ) {
		int depth = 0;
		while (S[i] == '-') {
			depth++;
			i++;
		}
		int nlen = 0;///数字长度
		while (nodeStack.size() > depth) {
			nodeStack.pop();
		}
		tmpNode = nodeStack.top();
		while (i + nlen < S.length() && S[i + nlen] != '-') {
			nlen++;
		}
		childNode = new TreeNode(stoi(S.substr(i, nlen)));
		tmpNode->left == nullptr ? tmpNode->left = childNode : tmpNode->right = childNode;
		nodeStack.push(childNode);
		i += nlen;
	}
	return ans;
}

///10. 正则表达式匹配
///动态规划，详情见leetcode题解
bool isMatch(string s, string p) {
	int m = s.size(), n = p.size();
	auto matches = [&](int i, int j) {
		if (i == 0) {
			return false;
		}
		if (p[j - 1] == '.') {
			return true;
		}
		return s[i - 1] == p[j - 1];
	};
	vector<vector<int>> f(m + 1, vector<int>(n + 1, 0));
	f[0][0] = 1;
	for (int i = 0; i <= m; i++) {
		for (int j = 1; j <= n; j++) {
			if (p[j - 1] == '*') {
				f[i][j] |= f[i][j - 2];
				if (matches(i, j - 1)) {
					f[i][j] |= f[i - 1][j];
				}
			}
			else {
				if (matches(i, j)) {
					f[i][j] |= f[i - 1][j - 1];
				}
			}
		}
	}
	return f[m][n];
}

///124. 二叉树中的最大路径和
int maxPathSum(TreeNode* root) {
	stack<TreeNode*> nodeStack;
	unordered_map<TreeNode*, int> maxpathMap;
	maxpathMap[nullptr] = 0;
	int ans = INT32_MIN;
	nodeStack.push(root);
	TreeNode* node;
	while (!nodeStack.empty()) {
		node = nodeStack.top();
		if (node->left || node->right) {
			bool bPopup = false;
			int tmpMax = node->val;
			if (node->left) {
				if (maxpathMap.count(node->left)) {
					tmpMax = max(maxpathMap[node->left] + node->val, tmpMax);
					bPopup = true;
				}
				else {
					nodeStack.push(node->left);
				}
			}
			if (node->right) {
				if (maxpathMap.count(node->right)) {
					tmpMax = max(maxpathMap[node->right] + node->val, tmpMax);
					bPopup = true;
				}
				else {
					nodeStack.push(node->right);
				}
			}
			if (bPopup) {
				maxpathMap[node] = tmpMax;
				ans = max(max(ans, maxpathMap[node]), node->val + maxpathMap[node->left] + maxpathMap[node->right]);
				nodeStack.pop();
			}
		}
		else {
			maxpathMap[node] = node->val;
			ans = max(ans, maxpathMap[node]);
			nodeStack.pop();
		}
	}
	return ans;
}

///面试题 16.18. 模式匹配
bool patternMatching(string pattern, string value) {
	int count_a = 0, count_b = 0;
	for (char ch : pattern) {
		if (ch == 'a') {
			++count_a;
		}
		else {
			++count_b;
		}
	}
	if (count_a < count_b) {
		swap(count_a, count_b);
		for (char& ch : pattern) {
			ch = (ch == 'a' ? 'b' : 'a');
		}
	}
	if (value.empty()) {
		return count_b == 0;
	}
	if (pattern.empty()) {
		return false;
	}
	for (int len_a = 0; count_a * len_a <= value.size(); ++len_a) {
		int rest = value.size() - count_a * len_a;
		if ((count_b == 0 && rest == 0) || (count_b != 0 && rest % count_b == 0)) {
			int len_b = (count_b == 0 ? 0 : rest / count_b);
			int pos = 0;
			bool correct = true;
			string value_a, value_b;
			for (char ch : pattern) {
				if (ch == 'a') {
					string sub = value.substr(pos, len_a);
					if (!value_a.size()) {
						value_a = move(sub);
					}
					else if (value_a != sub) {
						correct = false;
						break;
					}
					pos += len_a;
				}
				else {
					string sub = value.substr(pos, len_b);
					if (!value_b.size()) {
						value_b = move(sub);
					}
					else if (value_b != sub) {
						correct = false;
						break;
					}
					pos += len_b;
				}
			}
			if (correct && value_a != value_b) {
				return true;
			}
		}
	}
	return false;
}

///67. 二进制求和
string addBinary(string a, string b) {
	if (a.length() > b.length()) {
		swap(a, b);
	}
	int i = 0, k = 0;
	int m = a.length(), n = b.length();
	for (; i < m; i++) {
		int tmp = a[m - i - 1] - '0' + b[n - i - 1] - '0' + k;
		if (tmp < 2) {
			k = 0;
		}
		else if (tmp == 2) {
			k = 1;
			tmp = 0;
		}
		else if (tmp == 3) {
			k = 1;
			tmp = 1;
		}
		b[n - i - 1] = tmp + '0';
	}
	while (k > 0 && i < n) {
		int tmp = k + b[n - i - 1] - '0';
		k = 0;
		if (tmp == 2) {
			k = 1;
			tmp = 0;
		}
		b[n - i - 1] = tmp + '0';
		i++;
	}
	return k == 1 ? "1" + b : b;
}

//139. 单词拆分
//给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
//说明：
//拆分时可以重复使用字典中的单词。
//你可以假设字典中没有重复的单词。
//动态规划
bool wordBreak(string s, vector<string>& wordDict) {
	std::unordered_set<string> wordSet;
	for (size_t i = 0; i < wordDict.size(); i++) {
		wordSet.insert(wordDict[i]);
	}
	vector<bool> dp(s.size() + 1, false);
	dp[0] = true;
	for (size_t i = 1; i < s.size() + 1; i++) {
		for (int j = 0; j < i; j++) {
			if (dp[j] && wordSet.find(s.substr(j, i - j)) != wordSet.end()) {
				dp[i] = true;
				break;
			}
		}
	}
	return dp[s.size()];
}

//面试题 02.01. 移除重复节点
//编写代码，移除未排序链表中的重复节点。保留最开始出现的节点。
//示例1 :
//输入：[1, 2, 3, 3, 2, 1]
//输出：[1, 2, 3]
//示例2 :
//输入：[1, 1, 1, 1, 2]
//输出：[1, 2]
ListNode* removeDuplicateNodes(ListNode* head) {
	unordered_set<int> valSet;
	ListNode* p, * pnext;
	if (head) valSet.insert(head->val);
	else return head;
	p = head;
	pnext = p->next;
	while (pnext) {
		if (valSet.find(pnext->val) == valSet.end()) {
			valSet.insert(pnext->val);
			p = pnext;
			pnext = p->next;
		}
		else {
			p->next = pnext->next;
			delete pnext;
			pnext = p->next;
		}
	}
	return head;
}

//32. 最长有效括号
//给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。
//示例 1 :
//输入: "(()"
//输出 : 2
//解释 : 最长有效括号子串为 "()"
//示例 2 :
//输入 : ")()())"
//输出 : 4
//解释 : 最长有效括号子串为 "()()"
int longestValidParentheses(string s) {
	int nMax = 0;
	std::vector<int> dp(s.length(), 0);
	for (int i = 1; i < s.length(); i++) {
		switch (s[i]) {
		case '(':
			dp[i] = 0;
			break;
		case ')':
		{
			if (s[i - 1] == '(') {
				dp[i] = (i - 2 >= 0 ? dp[i - 2] : 0) + 2;
			}
			else {
				int n = i - dp[i - 1] - 1;
				if (n >= 0 && s[n] == '(')
					dp[i] = (n - 1 >= 0 ? dp[n - 1] : 0) + 2 + dp[i - 1];
			}
			break;
		}
		default:
			break;
		}
		nMax = dp[i] > nMax ? dp[i] : nMax;
	}
	return nMax;
}

//63. 不同路径 II
//一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
//机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
//现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
//网格中的障碍物和空位置分别用 1 和 0 来表示。
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
	if (obstacleGrid.size() == 0 || obstacleGrid[0].size() == 0 || obstacleGrid[0][0] == 1) {
		return 0;
	}
	int nRes = 0;
	int m = obstacleGrid.size(), n = obstacleGrid[0].size();

	/// <summary>
	/// 深度遍历，超时
	/// </summary>
	/// <param name="obstacleGrid"></param>
	/// <returns></returns>
	//struct localpos
	//{
	//	int x;
	//	int y;
	//	localpos(int m, int n) {
	//		x = m, y = n;
	//	}
	//};
	//std::stack<localpos> posStk;
	//posStk.push(localpos(0, 0));
	//while (!posStk.empty()) {
	//	localpos pos = posStk.top();
	//	posStk.pop();
	//	if (pos.x == m - 1 && pos.y == n - 1) {
	//		nRes++;
	//		continue;
	//	}
	//	if (pos.x + 1 < m && obstacleGrid[pos.x + 1][pos.y] == 0) {
	//		posStk.push(localpos(pos.x + 1, pos.y));
	//	}
	//	if (pos.y + 1 < n && obstacleGrid[pos.x][pos.y + 1] == 0) {
	//		posStk.push(localpos(pos.x, pos.y + 1));
	//	}
	//}

	/// <summary>
	/// 动态规划
	/// </summary>
	/// <param name="obstacleGrid"></param>
	/// <returns></returns>
	vector<vector<int>> dp(m, vector<int>(n, 0));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (obstacleGrid[i][j] == 1) dp[i][j] = 0;
			else if (!(i | j)) dp[i][j] = 1;
			else {
				dp[i][j] = (i - 1 >= 0 ? dp[i - 1][j] : 0) + (j - 1 >= 0 ? dp[i][j - 1] : 0);
			}
		}
	}
	nRes = dp[m - 1][n - 1];
	return nRes;
}

//49. 字母异位词分组
//给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
//示例 :
//输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
//输出 :
//	[
//		["ate", "eat", "tea"],
//		["nat", "tan"],
//		["bat"]
//	]
//说明：
//所有输入均为小写字母。
//不考虑答案输出的顺序。
vector<vector<string>> groupAnagrams(vector<string>& strs) {
	unordered_map<string, vector<string>> res;
	for (size_t i = 0; i < strs.size(); i++) {
		string tmp = strs[i];
		sort(tmp.begin(), tmp.end());
		res[tmp].push_back(strs[i]);
	}
	vector<vector<string>> ans;
	for (auto i = res.begin(); i != res.end(); i++) {
		ans.push_back(std::move(i->second));
	}
	return std::move(ans);
}

//51. N 皇后
//示例：
//输入：4
//输出： [
//	[".Q..",  // 解法 1
//		"...Q",
//		"Q...",
//		"..Q."],
//
//		["..Q.",  // 解法 2
//		"Q...",
//		"...Q",
//		".Q.."]
//]
//解释: 4 皇后问题存在两个不同的解法。
bool isOk(vector<string>& matrix, int m, int n) {
	for (int i = 0; i < m; i++) {
		if (matrix[i][n] == 'Q') return false;
	}
	for (int i = m - 1, j = n - 1; i >= 0 && j >= 0; i--, j--) {
		if (matrix[i][j] == 'Q') return false;
	}
	for (int i = m - 1, j = n + 1; i >= 0 && j < matrix.size(); i--, j++) {
		if (matrix[i][j] == 'Q') return false;
	}
	return true;
}
void placeQueens(vector<string>& matrix, vector<vector<string>>& ans, int m, int total) {
	for (int i = 0; i < total; i++) {
		matrix[m][i] = 'Q';
		if (isOk(matrix, m, i)) {
			if (m == total - 1) ans.push_back(matrix);
			else {
				placeQueens(matrix, ans, m + 1, total);
			}
		}
		matrix[m][i] = '.';
	}
}
vector<vector<string>> solveNQueens(int n) {
	vector<vector<string>> ans;
	vector<string> matrix(n, string(n, '.'));
	placeQueens(matrix, ans, 0, n);
	return ans;
}

//738. 单调递增的数字
//给定一个非负整数 N，找出小于或等于 N 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。
//（当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。）
//示例 1:
//输入: N = 10
//输出 : 9
//示例 2 :
//输入 : N = 1234
//输出 : 1234
//示例 3 :
//
//输入 : N = 332
//输出 : 299
//说明 : N 是在[0, 10 ^ 9] 范围内的一个整数。
int monotoneIncreasingDigits(int N) {
	//if (N < 10) return N;
	vector<int> vec;
	while (N) {
		vec.push_back(N % 10);
		N /= 10;
	}
	int i = vec.size() - 1;
	while (i > 0 && vec[i] <= vec[i - 1]) {
		i--;
	}
	while (i + 1 < vec.size() && vec[i] == vec[i + 1]) {
		i++;
	}
	if (i > 0) vec[i] -= 1;
	for (int j = i + 1; j < vec.size(); j++) {
		if (vec[j] <= vec[j - 1])break;
		else vec[j] -= 1;
	}
	int res = 0;
	for (int j = vec.size() - 1; j >= i; j--) {
		res = res * 10 + vec[j];
	}
	while (i > 0) {
		res = res * 10 + 9;
		i--;
	}
	return res;
}
//leetcode 290. 单词规律
bool wordPattern(string pattern, string s) {
	map<char, string> chrPattern;
	map<string, char> strPattern;
	int p = 0;
	istringstream ss(s);
	int i = 0;
	string tmp;
	while (i < pattern.length() && ss >> tmp) {
		auto it = chrPattern.find(pattern[i]);
		auto it2 = strPattern.find(tmp);
		if (it != chrPattern.end()) {
			if (tmp != it->second) return false;
		}
		else {
			chrPattern[pattern[i]] = tmp;
		}
		if (it2 != strPattern.end()) {
			if (pattern[i] != it2->second) return false;
		}
		else {
			strPattern[tmp] = pattern[i];
		}
		i++;
	}
	ss >> tmp;
	return i == pattern.size() && !ss;
}

//48. 旋转图像
//给定一个 n × n 的二维矩阵表示一个图像。
//将图像顺时针旋转 90 度。
//说明：
//你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。
//示例 1:
//给定 matrix =
//[
//	[1, 2, 3],
//	[4, 5, 6],
//	[7, 8, 9]
//],
//原地旋转输入矩阵，使其变为:
//[
//	[7, 4, 1],
//	[8, 5, 2],
//	[9, 6, 3]
//]
//示例 2:
//给定 matrix =
//[
//	[ 5, 1, 9, 11],
//	[2, 4, 8, 10],
//	[13, 3, 6, 7],
//	[15, 14, 12, 16]
//],
//原地旋转输入矩阵，使其变为:
//[
//	[15, 13, 2, 5],
//	[14, 3, 4, 1],
//	[12, 6, 8, 9],
//	[16, 7, 10, 11]
//]
void matrix_rotate(vector<vector<int>>& matrix, int m) {
	int n = matrix.size();
	int x1 = m, y1 = m;
	int x2 = n - x1 - 1, y2 = y1;
	int x3 = x2, y3 = x3;
	int x4 = x1, y4 = y3;
	for (; y1 < n - m - 1; y1++) {
		swap(matrix[x1][y1], matrix[x2][y2]);
		swap(matrix[x2][y2], matrix[x3][y3]);
		swap(matrix[x3][y3], matrix[x4][y4]);
		x2--;
		y3--;
		x4++;
	}
}
void matrix_rotate(vector<vector<int>>& matrix) {
	int n = matrix.size();
	int i = 0;
	while (i < n / 2) {
		matrix_rotate(matrix, i);
		i++;
	}
}

//103. 二叉树的锯齿形层序遍历
//给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
//例如：
//给定二叉树[3, 9, 20, null, null, 15, 7],
//3
/// \
//9  20
/// \
//15   7
//返回锯齿形层序遍历如下：
//
//[
//	[3],
//	[20, 9],
//	[15, 7]
//]
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
	bool flag = true;
	vector<vector<int>> ans;
	stack<TreeNode*> stk1;
	stack<TreeNode*> stk2;
	if (root) {
		stk1.push(root);
		while (!stk1.empty() || !stk2.empty()) {
			vector<int> tmp;
			if (flag) {
				while (!stk1.empty()) {
					TreeNode* node = stk1.top();
					stk1.pop();
					tmp.push_back(node->val);
					if (node->left) stk2.push(node->left);
					if (node->right) stk2.push(node->right);
				}
				flag = false;
			}
			else {
				while (!stk2.empty()) {
					TreeNode* node = stk2.top();
					stk2.pop();
					tmp.push_back(node->val);
					if (node->right) stk1.push(node->right);
					if (node->left) stk1.push(node->left);
				}
				flag = true;
			}
			ans.push_back(std::move(tmp));
		}
	}
	return std::move(ans);
}


int main()
{
	//vector<int> vec = { 8,1,5,2,6 };

	//cout << largestRectangleArea(vec) << endl;
	//vector<int> nums = { 100 };
	//cout << longestConsecutive(nums) << endl;
	//vector<int> nums = { 5,7,7,8,8,10 };
	//cout << shunxuSearch(nums, 10);
	//cout << reachingPoints(1, 1, 1000000000, 1);
	//cout << maxProfit(vec);
	//cout << maxScoreSightseeingPair(vec);
	//string S = "10-7--8";
	//TreeNode* root = recoverFromPreorder(S);

	//string s = "aab";
	//string p = "c*a*b";
	//cout << isMatch(s, p);

	//Codec codec;
	//string str = "[-10,9,20,null,null,15,7]";
	//TreeNode* root = codec.deserialize(str);
	//cout << maxPathSum(root);
	//cout << monotoneIncreasingDigits(668841);
	//string pattern = "abba", s = "dog cat cat dog";
	//wordPattern(pattern, s);

	vector<vector<int>> matrix = {
		{5, 1, 9,11},
		{2, 4, 8,10 },
		{13, 3, 6, 7},
		{15,14,12,16 } };
	matrix_rotate(matrix);
	for (size_t i = 0; i < matrix.size(); i++) {
		for (size_t j = 0; j < matrix.size(); j++) {
			std::cout << matrix[i][j] << ", ";
		}
		std::cout << std::endl;
	}

	return 0;
}
