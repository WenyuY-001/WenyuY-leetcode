using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;

namespace ConsoleAppTest;

public class Daily
{
    //1870、准时到达的列车最小时速（自己的，似了）
    public int MinSpeedOnTime1(int[] dist, double hour)
    {
        int n = dist.Length;
        if ((double)n - 1 >= hour)
            return -1;
        int[] dis = dist;
        int last = dist[n - 1];
        Array.Sort(dis);
        var e = hour - n + 1;
        if (e <= 0)
            return -1;
        var inte = CheckDecimalAndConvert(e);
        int speed = 0;
        if (n - inte < 0)
        {
            speed = dis[speed];
            int time = n + inte - 1;
            if (time > hour)
                speed++;
        }
        else
        {
            speed = dis[n - inte];
            int time = n + inte - 1;
            if (time > hour)
                speed = dis[n - inte + 1];
        }

        return speed;
    }

    int CheckDecimalAndConvert(double num)
    {
        if (num % 1 != 0)
            num += 1;
        return (int)num;
    }


    //1870、准时到达的列车最小时速（二分查找）
    const int FACTOR = 100;
    const int UPPER_BOUND = 10000000;

    public int MinSpeedOnTime(int[] dist, double hour)
    {
        long newHour = (long)Math.Round(hour * FACTOR);
        if (!CanArrive(dist, newHour, UPPER_BOUND))
        {
            return -1;
        }

        int low = 1, high = UPPER_BOUND;
        while (low < high)
        {
            int mid = low + (high - low) / 2;
            if (CanArrive(dist, newHour, mid))
            {
                high = mid;
            }
            else
            {
                low = mid + 1;
            }
        }

        return low;
    }

    bool CanArrive(int[] dist, long hour, int speed)
    {
        long totalTime = 0;
        int n = dist.Length;
        for (int i = 0; i < n; i++)
        {
            int time = i < n - 1 ? ((dist[i] - 1) / speed + 1) * FACTOR : (dist[i] * FACTOR - 1) / speed + 1;
            totalTime += time;
            if (totalTime > hour)
            {
                return false;
            }
        }

        return true;
    }


    //3101、交替子数组计数
    public long CountAlternatingSubArrays(int[] nums)
    {
        int n = nums.Length;
        long ans = 0; //关键long
        int n1 = nums[0];
        int count = 1;
        for (int i = 0; i < n; i++)
        {
            count = (n1 != nums[i]) ? ++count : 1;
            n1 = nums[i];
            ans += count;
        }

        return n + ans;
    }


    //1958、检查操作是否合法
    public bool CheckMove(char[][] board, int rMove, int cMove, char color)
    {
        if (board[rMove][cMove] != '.')
            return false;
        int[] dx = { 1, 1, 0, -1, -1, -1, 0, 1 }; // 行改变量
        int[] dy = { 0, 1, 1, 1, 0, -1, -1, -1 }; // 列改变量
        for (int k = 0; k < 8; ++k)
        {
            if (Check(board, rMove, cMove, color, dx[k], dy[k]))
                return true;
        }

        return false;
    }

    bool Check(char[][] board, int rMove, int cMove, char color, int dx, int dy)
    {
        int x = rMove + dx;
        int y = cMove + dy;
        int step = 1;
        while (x >= 0 && x < 8 && y >= 0 && y < 8)
        {
            if (step == 1)
            {
                if (board[x][y] == '.' || board[x][y] == color)
                    return false;
            }
            else
            {
                if (board[x][y] == '.')
                    return false;
                if (board[x][y] == color)
                    return true;
            }

            ++step;
            x += dx;
            y += dy;
        }

        return false;
    }


    //724、寻找数组的中心下标
    public int PivotIndex(int[] nums)
    {
        if (nums.Length == 0)
            return -1;

        int index = 0;
        int lSum = 0, rSum = nums.Sum() - nums[0];
        while (lSum < rSum && index < nums.Length - 1)
        {
            lSum += nums[index];
            index++;
            rSum -= nums[index + 1];
        }

        if (lSum == rSum)
            return index;
        else
            return -1;
    }


    //3102、最小化曼哈顿距离
    public int MinimumDistance(int[][] points)
    {
        //去除绝对值后，最大距离可以看出点的横纵坐标和的差和横纵坐标差的差中的较大值

        int[] arrS = new int[points.Length];
        int[] arrD = new int[points.Length];

        int sMax = int.MinValue;
        int sMax2 = int.MinValue;
        int sMin2 = int.MaxValue;
        int sMin = int.MaxValue;

        int dMax = int.MinValue;
        int dMax2 = int.MinValue;
        int dMin2 = int.MaxValue;
        int dMin = int.MaxValue;

        for (int i = 0; i < points.Length; i++)
        {
            arrS[i] = points[i][0] + points[i][1];
            arrD[i] = points[i][0] - points[i][1];

            if (arrS[i] > sMax)
            {
                sMax2 = sMax;
                sMax = arrS[i];
            }
            else if (arrS[i] > sMax2)
            {
                sMax2 = arrS[i];
            }

            if (arrS[i] < sMin)
            {
                sMin2 = sMin;
                sMin = arrS[i];
            }
            else if (arrS[i] < sMin2)
            {
                sMin2 = arrS[i];
            }

            if (arrD[i] > dMax)
            {
                dMax2 = dMax;
                dMax = arrD[i];
            }
            else if (arrD[i] > dMax2)
            {
                dMax2 = arrD[i];
            }

            if (arrD[i] < dMin)
            {
                dMin2 = dMin;
                dMin = arrD[i];
            }
            else if (arrD[i] < dMin2)
            {
                dMin2 = arrD[i];
            }
        }

        int ans = int.MaxValue;

        //判断排除的是否是最大值或最小值，如果是就使用次一级的值计算，否则就是最大值减去最小值
        for (int i = 0; i < points.Length; i++)
        {
            int maxS = 0;
            int maxD = 0;

            int c1 = arrS[i];
            int c2 = arrD[i];

            if (c1 == sMax)
                maxS = sMax2 - sMin;
            else if (c1 == sMin)
                maxS = sMax - sMin2;
            else
                maxS = sMax - sMin;

            if (c2 == dMax)
                maxD = dMax2 - dMin;
            else if (c2 == dMin)
                maxD = dMax - dMin2;
            else
                maxD = dMax - dMin;

            int max = Math.Max(maxS, maxD);
            ans = Math.Min(ans, max);
        }

        return ans;
    }

    // 2970、统计移除递增子数组的数目1
    public int IncremovableSubarrayCount1(int[] nums)
    {
        int left = 0, nl = nums.Length;
        while (left < nl - 1)
        {
            if (nums[left] >= nums[left + 1])
                break;
            left++;
        }

        if (left == nl - 1)
            return nl * (nl + 1) / 2;

        int ans = left + 2;
        for (int right = nl - 1; right > 0; right--)
        {
            if (right < nl - 1 && nums[right] >= nums[right + 1])
                break;

            while (left >= 0 && nums[left] >= nums[right])
                left--;
            ans += left + 2;
        }

        return ans;
    }


    //2974、最小数字游戏
    public int[] NumberGame(int[] nums)
    {
        int nl = nums.Length;
        int[] ans = new int[nl];
        Array.Copy(nums, ans, nl);
        int f;
        Array.Sort(ans);
        for (int i = 0; i < nl; i += 2)
        {
            f = ans[i];
            ans[i] = ans[i + 1];
            ans[i + 1] = f;
        }

        return ans;
    }


    //3011、判断一个数组是否可以变为有序
    public bool CanSortArray(int[] nums)
    {
        int length = nums.Length;
        int nextOnes = BitCount(nums[0]);
        int start = 0;
        for (int i = 0; i < length; i++)
        {
            int currOnes = nextOnes;
            nextOnes = i < length - 1 ? BitCount(nums[i + 1]) : 0;
            if (currOnes != nextOnes)
            {
                Array.Sort(nums, start, i - start + 1);
                start = i + 1;
            }
        }

        for (int i = 1; i < length; i++)
        {
            if (nums[i] < nums[i - 1])
            {
                return false;
            }
        }

        return true;
    }

    int BitCount(int num)
    {
        uint bits = (uint)num;
        bits = bits - ((bits >> 1) & 0x55555555);
        bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333);
        bits = (bits + (bits >> 4)) & 0x0f0f0f0f;
        bits = (bits + (bits >> 8)) & 0x00ff00ff;
        bits = (bits + (bits >> 16)) & 0x0000ffff;
        return (int)bits;
    }


    //807、保持城市天际线
    public int MaxIncreaseKeepingSkyline(int[][] grid)
    {
        int nl = grid.Length, sum = 0;
        int[] row = new int[nl], col = new int[nl];
        for (int i = 0; i < nl; ++i)
        {
            for (int k = 0; k < nl; ++k)
            {
                row[i] = Math.Max(row[i], grid[i][k]);
                col[k] = Math.Max(col[k], grid[i][k]);
            }
        }

        for (int i = 0; i < nl; ++i)
        {
            for (int k = 0; k < nl; ++k)
            {
                sum += Math.Min(row[i], col[k]) - grid[i][k];
            }
        }

        return sum;
    }


    //721、账户合并（一坨屎）
    public IList<IList<string>> AccountsMerge(IList<IList<string>> accounts)
    {
        for (int i = 0; i < accounts.Count; i++)
        {
            string name = accounts[i][0];
            List<string> listSet = new List<string>();
            for (int j = i + 1; j < accounts.Count; j++)
            {
                if (accounts[j].Contains(name))
                {
                    for (int k = 1; k < accounts[i].Count; k++)
                    {
                        if (accounts[j].Contains(accounts[i][k]))
                        {
                            HashSet<string> hashSet = new HashSet<string>();
                            foreach (var value in accounts[i])
                            {
                                hashSet.Add(value);
                            }

                            foreach (var value in accounts[j])
                            {
                                hashSet.Add(value);
                            }

                            accounts.Remove(accounts[j]);
                            listSet = hashSet.ToList();
                            listSet.Sort();
                            accounts[i].Clear();
                            accounts[i].Add(name);
                            foreach (var v in listSet)
                            {
                                accounts[i].Add(v);
                            }

                            break;
                        }
                    }
                }
            }
        }

        return accounts;
    }


    //2956、找到两个数组中的公共元素
    public int[] FindIntersectionValues(int[] nums1, int[] nums2)
    {
        int[] ans = new int[2];
        foreach (var i in nums1)
        {
            if (nums2.Contains(i))
            {
                ans[0]++;
            }
        }

        foreach (var i in nums2)
        {
            if (nums1.Contains(i))
            {
                ans[1]++;
            }
        }

        return ans;
    }


    //2959、关闭分部的可行集合数目（纯粘贴）
    public int NumberOfSets(int n, int maxDistance, int[][] roads)
    {
        int res = 0;
        int[] opened = new int[n];
        int[,] d = new int[n, n];

        for (int mask = 0; mask < (1 << n); mask++)
        {
            for (int i = 0; i < n; i++)
            {
                opened[i] = mask & (1 << i);
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    d[i, j] = 1000000;
                }
            }

            foreach (int[] road in roads)
            {
                int i = road[0], j = road[1], r = road[2];
                if (opened[i] > 0 && opened[j] > 0)
                {
                    d[i, j] = d[j, i] = Math.Min(d[i, j], r);
                }
            }

            // Floyd-Warshall algorithm
            for (int k = 0; k < n; k++)
            {
                if (opened[k] > 0)
                {
                    for (int i = 0; i < n; i++)
                    {
                        if (opened[i] > 0)
                        {
                            for (int j = i + 1; j < n; j++)
                            {
                                if (opened[j] > 0)
                                {
                                    d[i, j] = d[j, i] = Math.Min(d[i, j], d[i, k] + d[k, j]);
                                }
                            }
                        }
                    }
                }
            }

            // Validate
            int good = 1;
            for (int i = 0; i < n; i++)
            {
                if (opened[i] > 0)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (opened[j] > 0)
                        {
                            if (d[i, j] > maxDistance)
                            {
                                good = 0;
                                break;
                            }
                        }
                    }

                    if (good == 0)
                    {
                        break;
                    }
                }
            }

            res += good;
        }

        return res;
    }


    //3112、访问消失节点的最少时间
    public int[] MinimumTime(int n, int[][] edges, int[] disappear)
    {
        List<List<(int u, int v, int w)>> edges2 = [];
        for (int i = 0; i < n; i++)
        {
            edges2.Add([]);
        }

        foreach (int[] item in edges)
        {
            var (u, v, w) = (item[0], item[1], item[2]);
            edges2[u].Add((u, v, w));
            edges2[v].Add((v, u, w));
        }

        int[] dis = new int[n];
        Array.Fill(dis, int.MaxValue);
        dis[0] = 0;
        PriorityQueue<(int u, int v, int w), int> priorityQueue = new();
        priorityQueue.Enqueue((0, 0, 0), 0);

        while (priorityQueue.Count > 0)
        {
            var (_, x0, len0) = priorityQueue.Dequeue();
            if (len0 > dis[x0])
            {
                continue;
            }

            foreach (var (_, x1, len1) in edges2[x0])
            {
                if (dis[x0] + len1 < dis[x1] && dis[x0] + len1 < disappear[x1])
                {
                    dis[x1] = dis[x0] + len1;
                    priorityQueue.Enqueue((x0, x1, dis[x1]), dis[x1]);
                }
            }
        }

        for (int i = 0; i < dis.Length; i++)
        {
            if (dis[i] == int.MaxValue)
            {
                dis[i] = -1;
            }
        }

        return dis;
    }


    //3096、得到更多分数的最少关卡数目
    public int MinimumLevels(int[] possible)
    {
        int pre = 0, nl = possible.Length, tot = possible.Sum();
        for (int i = 0; i < nl - 1; ++i)
        {
            pre += possible[i];
            if (pre * 2 - i - 1 > tot * 2 - pre * 2 - nl + i + 1) return i + 1;
        }

        return -1;
    }


    //2850、将石头分散到网格图的最少移动次数
    public int MinimumMoves(int[][] grid)
    {
        // List<int[]> dis = new List<int[]>();
        // List<int[]> zeroPoint = new List<int[]>();
        // for (int i = 0; i < 3; i++)
        // {
        //     for (int j = 0; j < 3; j++)
        //     {
        //         if (grid[i][j] == 0)
        //         {
        //             zeroPoint.Add([i, j]);
        //         }
        //
        //         if (grid[i][j] > 1)
        //         {
        //             dis.Add([grid[i][j] - 1, i, j]);
        //         }
        //     }
        // }
        //
        // for (int i = 0; i < dis.Count; i++)
        // {
        //     dis[i]
        // }

        List<int[]> p1 = new List<int[]>();
        List<int[]> p2 = new List<int[]>();

        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                if (grid[r][c] > 1)
                {
                    for (int i = 0; i < grid[r][c] - 1; i++)
                    {
                        p1.Add([r, c]);
                    }
                }

                if (grid[r][c] == 0)
                {
                    p2.Add([r, c]);
                }
            }
        }

        if (p1.Count == 0) return 0;

        return DFS(0, (1 << p1.Count) - 1, p1, p2);
    }

    int DFS(int p, int msk, List<int[]> p1, List<int[]> p2)
    {
        Dictionary<(int, int), int> cache = new Dictionary<(int, int), int>();
        if (p == p1.Count) return 0;

        if (cache.ContainsKey((p, msk))) return cache[(p, msk)];

        int ans = 18;
        for (int j = 0; j < p2.Count; j++)
        {
            if ((msk & (1 << j)) != 0)
            {
                ans = Math.Min(ans,
                    Math.Abs(p1[p][0] - p2[j][0]) + Math.Abs(p1[p][1] - p2[j][1]) + DFS(p + 1, msk ^ (1 << j), p1, p2));
            }
        }

        cache[(p, msk)] = ans;
        return ans;
    }


    //1186、删除一次得到子数组最大和
    public int MaximumSum(int[] arr)
    {
        int dp0 = arr[0], dp1 = 0, res = arr[0];

        for (int i = 1; i < arr.Length; i++)
        {
            dp1 = Math.Max(dp0, dp1 + arr[i]);
            dp0 = Math.Max(dp0, 0) + arr[i];
            res = Math.Max(res, Math.Max(dp0, dp1));
        }

        return res;
    }


    //2101、引爆最多的炸弹
    public int MaximumDetonation(int[][] bombs)
    {
        int n = bombs.Length;
        // 维护引爆关系有向图
        IDictionary<int, IList<int>> edges = new Dictionary<int, IList<int>>();

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (i != j && IsConnected(bombs, i, j))
                {
                    edges.TryAdd(i, new List<int>());
                    edges[i].Add(j);
                }
            }
        }

        int res = 0; // 最多引爆数量

        for (int i = 0; i < n; ++i)
        {
            // 遍历每个炸弹，广度优先搜索计算该炸弹可引爆的数量，并维护最大值
            bool[] visited = new bool[n];
            int cnt = 1;
            Queue<int> queue = new Queue<int>();
            queue.Enqueue(i);
            visited[i] = true;
            while (queue.Count > 0)
            {
                int cidx = queue.Dequeue();
                foreach (int nidx in edges.ContainsKey(cidx) ? edges[cidx] : new List<int>())
                {
                    if (visited[nidx])
                    {
                        continue;
                    }

                    ++cnt;
                    queue.Enqueue(nidx);
                    visited[nidx] = true;
                }
            }

            res = Math.Max(res, cnt);
        }

        return res;
    }

    bool IsConnected(int[][] bombs, int u, int v)
    {
        long dx = bombs[u][0] - bombs[v][0];
        long dy = bombs[u][1] - bombs[v][1];
        return (long)bombs[u][2] * bombs[u][2] >= dx * dx + dy * dy;
    }


    //3098、求出所有子序列的能量和
    public int SumOfPowers(int[] nums, int k)
    {
        int MOD = 1000000007, INF = 0x3f3f3f3f;

        int n = nums.Length;
        Array.Sort(nums);
        ISet<int> set = new HashSet<int>();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                set.Add(nums[i] - nums[j]);
            }
        }

        set.Add(INF);
        IList<int> vals = new List<int>(set);
        ((List<int>)vals).Sort();

        int[][][] d = new int[n][][];

        for (int i = 0; i < n; i++)
        {
            d[i] = new int[k + 1][];
            for (int j = 0; j <= k; j++)
            {
                d[i][j] = new int[vals.Count];
            }
        }

        int[][] border = new int[n][];

        for (int i = 0; i < n; i++)
        {
            border[i] = new int[k + 1];
        }

        int[][] sum = new int[k + 1][];

        for (int i = 0; i <= k; i++)
        {
            sum[i] = new int[vals.Count];
        }

        int[][] suf = new int[n][];

        for (int i = 0; i < n; i++)
        {
            suf[i] = new int[k + 1];
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                int pos = BinarySearch(vals, nums[i] - nums[j]);

                for (int p = 1; p <= k; p++)
                {
                    while (border[j][p] < pos)
                    {
                        sum[p][border[j][p]] = (sum[p][border[j][p]] - suf[j][p] + MOD) % MOD;
                        sum[p][border[j][p]] = (sum[p][border[j][p]] + d[j][p][border[j][p]]) % MOD;
                        suf[j][p] = (suf[j][p] - d[j][p][border[j][p]] + MOD) % MOD;
                        border[j][p]++;
                        sum[p][border[j][p]] = (sum[p][border[j][p]] + suf[j][p]);
                    }
                }
            }

            d[i][1][vals.Count - 1] = 1;

            for (int p = 2; p <= k; p++)
            {
                for (int v = 0; v < vals.Count; v++)
                {
                    d[i][p][v] = sum[p - 1][v];
                }
            }

            for (int p = 1; p <= k; p++)
            {
                for (int v = 0; v < vals.Count; v++)
                {
                    suf[i][p] = (suf[i][p] + d[i][p][v]) % MOD;
                }

                sum[p][0] = (sum[p][0] + suf[i][p]) % MOD;
            }
        }

        int res = 0;

        for (int i = 0; i < n; i++)
        {
            for (int v = 0; v < vals.Count; v++)
            {
                res = (int)((res + 1L * vals[v] * d[i][k][v] % MOD) % MOD);
            }
        }

        return res;
    }

    int BinarySearch(IList<int> vals, int target)
    {
        int low = 0, high = vals.Count;
        while (low < high)
        {
            int mid = low + (high - low) / 2;
            if (vals[mid] >= target)
            {
                high = mid;
            }
            else
            {
                low = mid + 1;
            }
        }

        return low;
    }


    //2766、重新放置石块
    public IList<int> RelocateMarbles(int[] nums, int[] moveFrom, int[] moveTo)
    {
        var theSet = nums.ToHashSet();
        for (int i = 0; i < moveFrom.Length; i++)
        {
            theSet.Remove(moveFrom[i]);
            theSet.Add(moveTo[i]);
        }

        return theSet.Select(i => i).Order().ToList();
    }


    //2844、生成特殊数字的最少操作
    public int MinimumOperations(string num)
    {
        int nl = num.Length;
        bool has0 = false, has5 = false;

        for (int i = nl - 1; i >= 0; i--)
        {
            if (num[i] == '0' && !has0)
            {
                has0 = true;
            }
            else if (num[i] == '0' && has0)
            {
                return nl - i - 2;
            }
            else if (num[i] == '5' && has0)
            {
                return nl - i - 2;
            }
            else if (num[i] == '5' && !has0)
            {
                has5 = true;
            }
            else if (num[i] == '2' && has5)
            {
                return nl - i - 2;
            }
            else if (num[i] == '7' && has5)
            {
                return nl - i - 2;
            }
        }

        return has0 ? nl - 1 : nl;
    }


    //2740、找出分区值
    public int FindValueOfPartition(int[] nums) // 差值可以直接存在原数组里
    {
        int nl = nums.Length;
        int ans = Int32.MaxValue;
        Array.Sort(nums);
        for (int i = 0; i < nl - 1; i++)
        {
            ans = Math.Min(nums[i + 1] - nums[i], ans);
        }

        return ans;
    }


    //3106、满足距离约束且字典序最小的字符串
    public string GetSmallestString(string s, int k)
    {
        char[] stc = s.ToCharArray();
        for (int i = 0; i < s.Length; ++i)
        {
            int dis = Math.Min(s[i] - 'a', 'z' - s[i] + 1);
            if (dis <= k)
            {
                stc[i] = 'a';
                k -= dis;
            }
            else
            {
                stc[i] = (char)(stc[i] - k);
                break;
            }
        }

        return new string(stc);
    }


    //699、掉落的方块
    public IList<int> FallingSquares(int[][] positions)
    {
        int nl = positions.Length;
        int[] h = new int[nl];
        int[] ans = new int[nl];

        for (int i = 0; i < nl; i++)
        {
            int il = positions[i][0];
            int ir = positions[i][1] + il;
            for (int j = 0; j < i; j++)
            {
                int jl = positions[j][0];
                int jr = positions[j][1] + jl;
                if (ir > jl && jr > il)
                    h[i] = Math.Max(h[i], h[j]);
            }

            h[i] += positions[i][1];
            ans[i] = Math.Max(h[i], i != 0 ? ans[i - 1] : 0);
        }

        return ans;
    }


    //682、棒球比赛
    public int CalPoints(string[] operations)
    {
        int[] ans = new int[operations.Length];
        int i = 0;
        foreach (var s in operations)
        {
            if (s == "C")
            {
                ans[i - 1] = 0;
                i -= 2;
            }
            else if (s == "D")
            {
                ans[i] = ans[i - 1] * 2;
            }
            else if (s == "+")
            {
                ans[i] = ans[i - 1] + ans[i - 2];
            }
            else
            {
                ans[i] = Convert.ToInt32(s);
            }

            i++;
        }

        return ans.Sum();
    }


    //2961、双模幂运算
    public IList<int> GetGoodIndices(int[][] variables, int target)
    {
        IList<int> res = new List<int>();
        for (int i = 0; i < variables.Length; i++)
        {
            int[] ans = variables[i];
            int bAn = ans[0] % 10; // 只需取最后一位
            int exponent1 = ans[1];
            int exponent2 = ans[2];
            int mod = ans[3];

            // 使用快速幂计算 base^exponent1 % 10
            int lastDigit = FastPow(bAn, exponent1, 10);

            // 使用快速幂计算 lastDigit^exponent2 % mod
            int result = FastPow(lastDigit, exponent2, mod);

            if (result == target)
            {
                res.Add(i);
            }
        }

        return res;
    }

    // 快速幂方法
    private int FastPow(int di, int exp, int mod)
    {
        int result = 1;
        while (exp > 0)
        {
            if ((exp & 1) == 1)
            {
                // 如果exp是奇数
                result = (result * di) % mod;
            }

            di = di * di % mod;
            exp >>= 1; // exp /= 2
        }

        return result;
    }


    //3111、覆盖所有点的最少矩形数目
    public int MinRectanglesToCoverPoints(int[][] points, int w)
    {
        Dictionary<int, int> rects = new Dictionary<int, int>();
        int left = -1;
        int ans = 0;
        Array.Sort(points, (a, b) => a[0].CompareTo(b[0]));
        foreach (var p in points)
        {
            if (left == -1)
            {
                left = p[0];
                ans++;
                continue;
            }

            if (p[0] - left > w)
            {
                left = p[0];
                ans++;
            }
        }

        return ans;
    }


    //LCP 40、心算挑战
    public int MaxmiumScore(int[] cards, int cnt)
    {
        Array.Sort(cards);
        List<int> even = new(), odd = new();
        int e = 0, o = 0;
        even.Add(0);
        odd.Add(0);
        for (int i = cards.Length - 1; i >= 0; i--)
        {
            if ((cards[i] & 1) == 0)
            {
                e += cards[i];
                even.Add(e);
            }
            else
            {
                o += cards[i];
                odd.Add(o);
            }
        }

        int elen = even.Count() - 1, olen = odd.Count - 1;
        o = Math.Min((olen & 1) == 0 ? olen : olen - 1, (cnt & 1) == 0 ? cnt : cnt - 1);
        e = cnt - o;
        if (e > elen) return 0;
        int m = 0;
        while (e <= elen && o >= 0)
        {
            int n = even[e] + odd[o];
            m = Math.Max(m, n);
            o -= 2;
            e += 2;
        }

        return m;
    }


    //3128、直角三角形
    public long NumberOfRightTriangles(int[][] grid)
    {
        int n = grid.Length, m = grid[0].Length;
        int[] col = new int[m];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                col[j] += grid[i][j];
            }
        }

        long ans = 0;

        for (int i = 0; i < n; i++)
        {
            int row = grid[i].Sum();
            for (int j = 0; j < m; j++)
            {
                if (grid[i][j] == 1)
                {
                    ans += (row - 1) * (col[j] - 1);
                }
            }
        }

        return ans;
    }


    //3143、正方形中的最多点数
    public int MaxPointsInsideSquare(int[][] points, string s)
    {
        int[] arr = new int[26];
        Array.Fill(arr, int.MaxValue);
        int min = arr[0];
        for (int i = 0, len = points.Length; i < len; ++i)
        {
            int[] p = points[i];
            int idx = s[i] - 'a',
                d = Math.Max(Math.Abs(p[0]), Math.Abs(p[1]));
            if (d < arr[idx])
            {
                min = Math.Min(min, arr[idx]);
                arr[idx] = d;
            }
            else if (d < min)
                min = d;
        }

        int ans = 0;

        foreach (int i in arr)
            if (i < min)
                ++ans;

        return ans;
    }


    //572、另一棵树的子树
    public bool IsSubtree(TreeNode root, TreeNode subRoot)
    {
        return Dfs(root, subRoot);
    }

    public bool Dfs(TreeNode root, TreeNode subRoot)
    {
        if (root == null)
        {
            return false;
        }

        return Check(root, subRoot) || Dfs(root.left, subRoot) || Dfs(root.right, subRoot);
    }

    public bool Check(TreeNode root, TreeNode subRoot)
    {
        if (root == null && subRoot == null)
        {
            return true;
        }

        if (root == null || subRoot == null || root.val != subRoot.val)
        {
            return false;
        }

        return Check(root.left, subRoot.left) && Check(root.right, subRoot.right);
    }


    //600、不含连续1的非负整数
    public int FindIntegers(int n)
    {
        //遍历二进制字符串来计算ans（特殊情况太多）
        string its = Convert.ToString(n, 2);
        int ans = n + 1, nl = its.Length;
        int[][] mn = new int[nl + 1][];
        if (nl < 3)
        {
            if (n == 3)
                return 3;
            return ans;
        }

        mn[0] = [0, 0, 0];
        mn[1] = [1, 1, 1];
        mn[2] = [2, 3, 3];
        for (int i = 3; i <= nl; i++)
        {
            int sum = (int)Math.Pow(2, i - 1) + mn[i - 2][2];
            mn[i] = [sum, mn[i - 1][1] * 2, mn[i - 1][2] + sum];
        }

        ans -= mn[nl - 2][2];
        char[] chars = new char[nl];
        Array.Fill(chars, '0');
        chars[0] = '1';

        for (int i = 1; i < nl - 1; i++)
        {
            chars[i] = its[i];
            if (its[i] == '1' && its[i - 1] == '1')
            {
                ans--;
                ans -= mn[nl - i - 2][2];
                ans -= n - Convert.ToInt32(new string(chars), 2);
                return ans;
            }

            if (its[i] == '1' && i < nl - 2)
            {
                ans -= mn[nl - i - 2][2];
            }
        }

        if (its[nl - 1] == '1' && its[nl - 2] == '1')
        {
            ans--;
        }

        return ans;


        // 11 1
        // 110 111 2
        // 1011   1100 1101 1110 1111 5 1 4
        // 10011 10110 10111   11000 11001 11010 11011 11100 11101 11110 11111 11 3 8
        // 100011 100110 100111 101011 101100 101101 101110 101111   110000 110001 110010 110011 110100 110101 110110 110111 111000 111001 111010 111011 111100 111101 111110 111111 24 8 16
        // 1000011 1000110 1000111 1001011 1001100  1001101 1001110 1001111 1010011 1010110  1010111 1011000 1011001 1011010 1011011  1011100 1011101 1011110 1011111
        //     19 32

        // int[] dp = new int[31];
        // dp[0] = dp[1] = 1;
        // for (int i = 2; i < 31; ++i) {
        //     dp[i] = dp[i - 1] + dp[i - 2];
        // }
        //
        // int pre = 0, ans = 0;
        // for (int i = 29; i >= 0; --i) {
        //     int val = 1 << i;
        //     if ((n & val) != 0) {
        //         ans += dp[i + 1];
        //         if (pre == 1) {
        //             break;
        //         }
        //         pre = 1;
        //     } else {
        //         pre = 0;
        //     }
        //
        //     if (i == 0) {
        //         ++ans;
        //     }
        // }
        //
        // return ans;
    }


    //3129、找出所有稳定的二进制数组1
    public int NumberOfStableArrays1(int zero, int one, int limit)
    {
        const long MOD = 1000000007;
        long[][][] dp = new long[zero + 1][][];
        for (int i = 0; i <= zero; i++)
        {
            dp[i] = new long[one + 1][];
            for (int j = 0; j <= one; j++)
            {
                dp[i][j] = new long[2];
            }
        }

        for (int i = 0; i <= Math.Min(zero, limit); i++)
        {
            dp[i][0][0] = 1;
        }

        for (int j = 0; j <= Math.Min(one, limit); j++)
        {
            dp[0][j][1] = 1;
        }

        for (int i = 1; i <= zero; i++)
        {
            for (int j = 1; j <= one; j++)
            {
                if (i > limit)
                {
                    dp[i][j][0] = dp[i - 1][j][0] + dp[i - 1][j][1] - dp[i - limit - 1][j][1];
                }
                else
                {
                    dp[i][j][0] = dp[i - 1][j][0] + dp[i - 1][j][1];
                }

                dp[i][j][0] = (dp[i][j][0] % MOD + MOD) % MOD;
                if (j > limit)
                {
                    dp[i][j][1] = dp[i][j - 1][1] + dp[i][j - 1][0] - dp[i][j - limit - 1][0];
                }
                else
                {
                    dp[i][j][1] = dp[i][j - 1][1] + dp[i][j - 1][0];
                }

                dp[i][j][1] = (dp[i][j][1] % MOD + MOD) % MOD;
            }
        }

        return (int)((dp[zero][one][0] + dp[zero][one][1]) % MOD);
    }


    //3130、找出所有稳定的二进制数组2
    public int NumberOfStableArrays2(int zero, int one, int limit)
    {
        const int MOD = 1000000007;
        int[][][] dp = new int[zero + 1][][];
        for (int i = 0; i <= zero; i++)
        {
            dp[i] = new int[one + 1][];
            for (int j = 0; j <= one; j++)
            {
                dp[i][j] = new int[2];
                for (int lastBit = 0; lastBit <= 1; lastBit++)
                {
                    if (i == 0)
                    {
                        if (lastBit == 0 || j > limit)
                        {
                            dp[i][j][lastBit] = 0;
                        }
                        else
                        {
                            dp[i][j][lastBit] = 1;
                        }
                    }
                    else if (j == 0)
                    {
                        if (lastBit == 1 || i > limit)
                        {
                            dp[i][j][lastBit] = 0;
                        }
                        else
                        {
                            dp[i][j][lastBit] = 1;
                        }
                    }
                    else if (lastBit == 0)
                    {
                        dp[i][j][lastBit] = dp[i - 1][j][0] + dp[i - 1][j][1];
                        if (i > limit)
                        {
                            dp[i][j][lastBit] -= dp[i - limit - 1][j][1];
                        }
                    }
                    else
                    {
                        dp[i][j][lastBit] = dp[i][j - 1][0] + dp[i][j - 1][1];
                        if (j > limit)
                        {
                            dp[i][j][lastBit] -= dp[i][j - limit - 1][0];
                        }
                    }

                    dp[i][j][lastBit] %= MOD;
                    if (dp[i][j][lastBit] < 0)
                    {
                        dp[i][j][lastBit] += MOD;
                    }
                }
            }
        }

        return (dp[zero][one][0] + dp[zero][one][1]) % MOD;
    }


    //3131、找出与数组相加的整数1
    public int AddedInteger(int[] nums1, int[] nums2)
    {
        return nums1.Min() - nums2.Min();
    }


    //3132、找出与数组相加的整数2
    public int MinimumAddedInteger(int[] nums1, int[] nums2)
    {
        int m = nums1.Length, n = nums2.Length;
        Array.Sort(nums1);
        Array.Sort(nums2);
        foreach (int i in new int[] { 2, 1, 0 })
        {
            int left = i + 1, right = 1;
            while (left < m && right < n)
            {
                if (nums1[left] - nums2[right] == nums1[i] - nums2[0])
                {
                    ++right;
                }

                ++left;
            }

            if (right == n)
            {
                return nums2[0] - nums1[i];
            }
        }

        return 0;
    }


    //2940、找到Alice和Bob可以相遇的建筑
    public int[] LeftmostBuildingQueries(int[] heights, int[][] queries)
    {
        int n = heights.Length;
        int m = queries.Length;
        IList<Tuple<int, int>>[] query = new IList<Tuple<int, int>>[n];
        for (int i = 0; i < n; i++)
        {
            query[i] = new List<Tuple<int, int>>();
        }

        int[] ans = new int[m];
        IList<int> st = new List<int>();

        for (int i = 0; i < m; i++)
        {
            int a = queries[i][0];
            int b = queries[i][1];
            if (a > b)
            {
                (a, b) = (b, a);
            }

            if (a == b || heights[a] < heights[b])
            {
                ans[i] = b;
                continue;
            }

            query[b].Add(new Tuple<int, int>(i, heights[a]));
        }

        int top = -1;
        for (int i = n - 1; i >= 0; i--)
        {
            for (int j = 0; j < query[i].Count; j++)
            {
                int q = query[i][j].Item1;
                int val = query[i][j].Item2;
                if (top == -1 || heights[st[0]] <= val)
                {
                    ans[q] = -1;
                    continue;
                }

                int l = 0, r = top;
                while (l <= r)
                {
                    int mid = (l + r) >> 1;
                    if (heights[st[mid]] > val)
                    {
                        l = mid + 1;
                    }
                    else
                    {
                        r = mid - 1;
                    }
                }

                ans[q] = st[r];
            }

            while (top >= 0 && heights[st[top]] <= heights[i])
            {
                st.RemoveAt(st.Count - 1);
                top--;
            }

            st.Add(i);
            top++;
        }

        return ans;
    }


    //1035、不相交的线
    public int MaxUncrossedLines(int[] nums1, int[] nums2)
    {
        int m = nums1.Length, n = nums2.Length;
        int[,] dp = new int[m + 1, n + 1];

        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                if (nums1[i - 1] == nums2[j - 1])
                {
                    dp[i, j] = dp[i - 1, j - 1] + 1;
                }
                else
                {
                    dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
                }
            }
        }

        return dp[m, n];
    }


    //3148、矩阵中的最大得分
    public int MaxScore(IList<IList<int>> grid)
    {
        int nl = grid.Count, ml = grid[0].Count;
        int[,] pre = new int[nl, ml];
        int ans = 0;
        for (int i = 0; i < nl; i++)
        {
            for (int j = 0; j < ml; j++)
            {
            }
        }

        return ans;
    }


    //3117、划分数组得到最小的值之和（？？？？）
    public int MinimumValueSum(int[] nums, int[] andValues)
    {
        var cache = new Dictionary<long, int>();
        int n = nums.Count();
        int m = andValues.Count();

        int DFS(int i, int j, int and)
        {
            if (m - j > n - i)
            {
                return int.MaxValue / 2;
            }

            if (j == m)
            {
                if (i == n)
                    return 0;
                else
                    return int.MaxValue / 2;
            }

            and &= nums[i];
            long mask = (long)i << 36 | (long)j << 32 |
                        (uint)and;
            if (cache.ContainsKey(mask))
                return cache[mask];
            int ret = DFS(i + 1, j, and);
            if (and == andValues[j])
            {
                ret = Math.Min(ret, DFS(i + 1, j + 1, -1) + nums[i]);
            }

            cache[mask] = ret;
            return ret;
        }

        int res = DFS(0, 0, -1);
        if (res < int.MaxValue / 2)
            return res;
        else
            return -1;
    }


    //3137、K周期字符串需要的最少操作次数
    public int MinimumOperationsToMakeKPeriodic(string word, int k)
    {
        int n = word.Length, res = int.MaxValue;
        Dictionary<string, int> count = new Dictionary<string, int>();

        for (int i = 0; i < n; i += k)
        {
            string part = word.Substring(i, k);
            count.TryAdd(part, 0);
            count[part]++;
            res = Math.Min(res, n / k - count[part]);
        }

        return res;
    }


    //551、学生出勤记录1
    public bool CheckRecord1(string s)
    {
        int aCount = 0, lCount = 0;
        foreach (var c in s)
        {
            if (c == 'A')
                aCount++;

            if (c == 'L')
                lCount++;
            else
                lCount = 0;

            if (aCount == 2)
                return false;

            if (lCount == 3)
                return false;
        }

        return true;
    }


    //552、学生出勤记录2
    public int CheckRecord(int n)
    {
        long pa = 0, la = 0, lla = 0, p = 1, l = 0, ll = 0;
        for (int i = 0; i < n; ++i)
        {
            long newPa = (p + l + ll + pa + la + lla) % 1000000007;
            long newLa = pa;
            long newLla = la;
            long newP = (p + l + ll) % 1000000007;
            long newL = p;
            long newLl = l;
            (pa, la, lla, p, l, ll) = (newPa, newLa, newLla, newP, newL, newLl);
        }

        return (int)((pa + la + lla + p + l + ll) % 1000000007);
    }


    //3154、到达第K级台阶的方案数
    public int WaysToReachStair(int k)
    {
        int n = 0, npow = 1, ans = 0;
        while (true)
        {
            if (npow - n - 1 <= k && k <= npow)
            {
                ans += Comb(n + 1, npow - k);
            }
            else if (npow - n - 1 > k)
            {
                break;
            }

            ++n;
            npow *= 2;
        }

        return ans;
    }

    int Comb(int n, int k)
    {
        long ans = 1;
        for (int i = n; i >= n - k + 1; --i)
        {
            ans *= i;
            ans /= n - i + 1;
        }

        return (int)ans;
    }


    //3007、价值和小于等于K的最大数字
    public long FindMaximumNumber(long k, int x)
    {
        long l = 1, r = (k + 1) << x;
        while (l < r)
        {
            long m = (l + r + 1) / 2;
            if (AccumulatedPrice(m) > k) r = m - 1;
            else l = m;
        }

        return l;

        long AccumulatedPrice(long num)
        {
            long res = 0, temp = num;
            int length = 0;
            while (temp > 0)
            {
                ++length;
                temp >>= 1;
            }

            for (int i = x; i <= length; i += x)
            {
                res += AccumulatedBitPrice(i, num);
            }

            return res;
        }

        long AccumulatedBitPrice(int x, long num)
        {
            long period = 1L << x;
            long res = period / 2 * (num / period);
            if (num % period >= period / 2)
            {
                res += num % period - (period / 2 - 1);
            }

            return res;
        }
    }


    //3133、数组最后一个元素的最小值
    public long MinEnd(int n, int x)
    {
        long end = n - 1;

        for (int i = 0; i < 30; ++i)
        {
            if ((x & (1 << i)) != 0)
            {
                end = (end >> i << (i + 1)) | (end & ((1 << i) - 1)) | (1 << i);
            }
        }

        return end;
    }


    //3145、大数组元素的乘积
    public int[] FindProductsOfElements(long[][] queries)
    {
        int[] ans = new int[queries.Length];

        for (int i = 0; i < queries.Length; i++)
        {
            // 偏移让数组下标从1开始
            queries[i][0]++;
            queries[i][1]++;
            long l = MidCheck(queries[i][0]);
            long r = MidCheck(queries[i][1]);
            int mod = (int)queries[i][2];

            long res = 1;
            long pre = CountOne(l - 1);
            for (int j = 0; j < 60; j++)
            {
                if ((1L << j & l) != 0)
                {
                    pre++;
                    if (pre >= queries[i][0] && pre <= queries[i][1])
                    {
                        res = res * (1L << j) % mod;
                    }
                }
            }

            if (r > l)
            {
                long bac = CountOne(r - 1);
                for (int j = 0; j < 60; j++)
                {
                    if ((1L << j & r) != 0)
                    {
                        bac++;
                        if (bac >= queries[i][0] && bac <= queries[i][1])
                        {
                            res = res * (1L << j) % mod;
                        }
                    }
                }
            }

            if (r - l > 1)
            {
                long xs = CountPow(r - 1) - CountPow(l);
                res = res * PowMod(2L, xs, mod) % mod;
            }

            ans[i] = (int)res;
        }

        return ans;

        long MidCheck(long x)
        {
            long l = 1, r = (long)1e15;
            while (l < r)
            {
                long mid = (l + r) >> 1;
                if (CountOne(mid) >= x)
                {
                    r = mid;
                }
                else
                {
                    l = mid + 1;
                }
            }

            return r;
        }

        // 计算 <= x 所有数的数位1的和
        long CountOne(long x)
        {
            long res = 0;
            int sum = 0;

            for (int i = 60; i >= 0; i--)
            {
                if ((1L << i & x) != 0)
                {
                    res += 1L * sum * (1L << i);
                    sum += 1;

                    if (i > 0)
                    {
                        res += 1L * i * (1L << (i - 1));
                    }
                }
            }

            res += sum;
            return res;
        }

        // 计算 <= x 所有数的数位对幂的贡献之和
        long CountPow(long x)
        {
            long res = 0;
            int sum = 0;

            for (int i = 60; i >= 0; i--)
            {
                if ((1L << i & x) != 0)
                {
                    res += 1L * sum * (1L << i);
                    sum += i;

                    if (i > 0)
                    {
                        res += 1L * i * (i - 1) / 2 * (1L << (i - 1));
                    }
                }
            }

            res += sum;
            return res;
        }

        int PowMod(long x, long y, int mod)
        {
            long res = 1;
            while (y != 0)
            {
                if ((y & 1) != 0)
                {
                    res = res * x % mod;
                }

                x = x * x % mod;
                y >>= 1;
            }

            return (int)res;
        }
    }


    //3146、两个字符串的排列差
    public int FindPermutationDifference(string s, string t)
    {
        int ans = 0;

        for (int i = 0; i < s.Length; i++)
        {
            ans += Math.Abs(t.IndexOf(s[i]) - i);
        }

        return ans;
    }


    //698、划分为K个相等的子集
    public bool CanPartitionKSubsets(int[] nums, int k)
    {
        int sum = nums.Sum();

        if (sum % k != 0)
            return false;

        int per = sum / k;
        Array.Sort(nums);
        int n = nums.Length;

        if (nums[n - 1] > per)
            return false;


        bool[] dp = new bool[1 << n];
        int[] curSum = new int[1 << n];
        dp[0] = true;

        for (int i = 0; i < 1 << n; i++)
        {
            if (!dp[i])
                continue;

            for (int j = 0; j < n; j++)
            {
                if (curSum[i] + nums[j] > per)
                    break;

                if (((i >> j) & 1) == 0)
                {
                    int next = i | (1 << j);
                    if (!dp[next])
                    {
                        curSum[next] = (curSum[i] + nums[j]) % per;
                        dp[next] = true;
                    }
                }
            }
        }

        return dp[(1 << n) - 1];
    }


    //690、员工的重要性
    public int GetImportance(IList<Employee> employees, int id)
    {
        IList<int> IDindex = new List<int>();
        foreach (var employee in employees)
        {
            IDindex.Add(employee.id);
        }

        int ans = 0;

        AddIm(employees[IDindex.IndexOf(id)]);

        return ans;


        void AddIm(Employee employee)
        {
            ans += employee.importance;
            if (employee.subordinates.Count != 0)
            {
                foreach (var i in employee.subordinates)
                {
                    AddIm(employees[IDindex.IndexOf(i)]);
                }
            }
        }
    }


    //3134、找出唯一性数组的中位数
    public int MedianOfUniquenessArray(int[] nums)
    {
        int n = nums.Length;
        long median = ((long)n * (n + 1) / 2 + 1) / 2;
        int res = 0;
        int lo = 1, hi = n;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (Check(nums, mid, median))
            {
                res = mid;
                hi = mid - 1;
            }
            else
            {
                lo = mid + 1;
            }
        }

        return res;

        bool Check(int[] nums, int t, long median)
        {
            Dictionary<int, int> cnt = new Dictionary<int, int>();
            long tot = 0;
            for (int i = 0, j = 0; i < nums.Length; i++)
            {
                if (cnt.ContainsKey(nums[i]))
                {
                    cnt[nums[i]]++;
                }
                else
                {
                    cnt[nums[i]] = 1;
                }

                while (cnt.Count > t)
                {
                    cnt[nums[j]]--;
                    if (cnt[nums[j]] == 0)
                    {
                        cnt.Remove(nums[j]);
                    }

                    j++;
                }

                tot += i - j + 1;
            }

            return tot >= median;
        }
    }


    //3144、分割字符频率相等的最少子字符串
    public int MinimumSubstringsInPartition(string s)
    {
        int n = s.Length;
        int[] dp = new int[n + 1];

        for (int i = 1; i <= n; ++i)
        {
            dp[i] = dp[i - 1] + 1;
            int max = 1;

            IDictionary<char, int> dic = new Dictionary<char, int>();
            dic.Add(s[i - 1], 1);

            for (int j = i - 1; j > 0; --j)
            {
                char c = s[j - 1];
                dic.TryAdd(c, 0);
                max = Math.Max(max, ++dic[c]);

                if (max * dic.Count == i - j + 1)
                    dp[i] = Math.Min(dp[i], dp[j - 1] + 1);
            }
        }

        return dp[n];
    }


    //3142、判断矩阵是否满足条件
    public bool SatisfiesConditions(int[][] grid)
    {
        int nl = grid[0].Length;

        for (int i = 0; i < nl - 1; i++)
        {
            if (grid[0][i] == grid[0][i + 1])
                return false;
        }

        for (int i = 1; i < grid.Length; i++)
        {
            for (int j = 0; j < nl; j++)
            {
                if (grid[i][j] != grid[0][j])
                    return false;
            }
        }

        return true;
    }


    //3153、所有数对中数位不同之和
    public long SumDigitDifferences(int[] nums)
    {
        long ans = 0;
        int n = nums.Length;
        while (nums[0] > 0)
        {
            int[] cnt = new int[10];
            for (int i = 0; i < n; i++)
            {
                cnt[nums[i] % 10]++;
                nums[i] /= 10;
            }

            for (int i = 0; i < 10; i++)
            {
                ans += (long)(n - cnt[i]) * cnt[i];
            }
        }

        return ans / 2;
    }
    
    
    //3127、构造相同颜色的正方形
    public bool CanMakeSquare(char[][] grid)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                int[] bw = new int[2];
                if (grid[i][j] == 'B')
                    bw[0]++;
                else
                    bw[1]++;
                if (grid[i + 1][j] == 'B')
                    bw[0]++;
                else
                    bw[1]++;
                if (grid[i][j + 1] == 'B')
                    bw[0]++;
                else
                    bw[1]++;
                if (grid[i + 1][j + 1] == 'B')
                    bw[0]++;
                else
                    bw[1]++;
                if (bw[0] >= 3 || bw[1] >= 3)
                    return true;
            }
        }

        return false;
    }
    
    
    //1450、在既定时间做作业的学生人数
    public int BusyStudent(int[] startTime, int[] endTime, int queryTime)
    {
        int ans = 0;
        for (int i = 0; i < startTime.Length; i++)
        {
            if (startTime[i] <= queryTime)
                if (endTime[i] >= queryTime)
                    ans++;
        }

        return ans;
    }
    
    
    //2024、考试的最大困扰度
    public int MaxConsecutiveAnswers(string answerKey, int k) {
        int n = answerKey.Length;
        int ans = 0, cntT = 0;
        
        for(int i = 0, j = 0; j < n; ++j)
        {
            cntT += answerKey[j] == 'T' ? 1 : 0;
            
            while(i < j && cntT > k && (j - i + 1 - cntT) > k)
            {
                cntT -= answerKey[i++] == 'T' ? 1 : 0;
            }
            
            ans = Math.Max(ans, j - i + 1);
        }
 
        return ans;
    }
    
    
    //2708、一个小组的最大实力值
    public long MaxStrength(int[] nums)
    {
        long ans = 1;
        bool isChange = false;
        int negcnt = 0;
        int[] negatives = new int[12];
        foreach (var i in nums)
        {
            if (i < 0)
            {
                negatives[negcnt] = i;
                negcnt++;
            }
            else if (i == 0)
                continue;
            else
            {
                ans *= i;
                isChange = true;
            }
        }

        if (negcnt > 1)
        {
            isChange = true;
            Array.Sort(negatives);
            negcnt -= negcnt % 2;
            int cnt = 0;
            while (cnt < negcnt)
            {
                ans *= negatives[cnt];
                cnt++;
            }
        }

        if (!isChange)
            return nums.Max();

        return ans;
    }
    
    
    //2860、让所有学生保持开心的分组方法数
    public int CountWays(IList<int> nums)
    {
        int ans = 0;
        var nl = nums.Count;
        var cur = nums.ToArray();
        Array.Sort(cur);
        int cnt = 0;
        if (cur[0] > cnt)
            ans++;
        
        for (int i = 0; i < nl; i++)
        {
            cnt++;
            if (i == nl - 1)
                if (cnt > cur[i])
                {
                    ans++;
                    continue;
                }
            
            if (cur[i] < cnt && cur[i + 1] > cnt)
                ans++;
        }

        return ans;
        
        /*int count = 0;
        int n = nums.Count;
        ((List<int>) nums).Sort();
        for(int k =0;k<=n;k++){
            if(k>0&&nums[k-1]>=k){
                continue;
            }
            if(k<n&&nums[k]<=k){
                continue;
            }
            count++;
        }
        
        return count;*/
    }
    
    
    //3174、清除数字
    public string ClearDigits(string s)
    {
        char[] chars = new char[100];
        int index = 0;
        foreach (var c in s)
        {
            if (c > 47 && c < 58)
            {
                index--;
                chars[index] = ' ';
            }
            else
            {
                chars[index] = c;
                index++;
            }
        }

        return new string(chars).Replace("\u0000", "").Trim();
    }
    
    
    //3176、求出最长好子序列 I
    public int MaximumLength(int[] nums, int k)
    {
        int nl = nums.Length;
        IDictionary<int, int[]> dp = new Dictionary<int, int[]>();
        int[] zd = new int[k + 1];

        for (int i = 0; i < nl; i++) {
            int v = nums[i];
            dp.TryAdd(v, new int[k + 1]);

            int[] tmp = dp[v];
            for (int j = 0; j <= k; j++) {
                tmp[j] += 1;
                if (j > 0) {
                    tmp[j] = Math.Max(tmp[j], zd[j - 1] + 1);
                }
            }
            for (int j = 0; j <= k; j++) {
                zd[j] = Math.Max(zd[j], tmp[j]);
                if (j > 0) {
                    zd[j] = Math.Max(zd[j], zd[j - 1]);
                }
            }
        }
        
        return zd[k];
    }
}

//676、实现一个魔法字典
public class MagicDictionary
{
    private string[] words;

    public MagicDictionary()
    {
    }

    public void BuildDict(string[] dictionary)
    {
        words = dictionary;
    }

    public bool Search(string searchWord)
    {
        foreach (string word in words)
        {
            if (word.Length != searchWord.Length)
            {
                continue;
            }

            int diff = 0;
            for (int i = 0; i < word.Length; ++i)
            {
                if (word[i] != searchWord[i])
                {
                    ++diff;
                    if (diff > 1)
                    {
                        break;
                    }
                }
            }

            if (diff == 1)
                return true;
        }

        return false;
    }
}