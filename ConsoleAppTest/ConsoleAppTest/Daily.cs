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
    public IList<IList<string>> AccountsMerge(IList<IList<string>> accounts) {
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
    public int NumberOfSets(int n, int maxDistance, int[][] roads) {
        int res = 0;
        int[] opened = new int[n];
        int[,] d = new int[n, n];

        for (int mask = 0; mask < (1 << n); mask++) {
            for (int i = 0; i < n; i++) {
                opened[i] = mask & (1 << i);
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    d[i, j] = 1000000;
                }
            }
            foreach (int[] road in roads) {
                int i = road[0], j = road[1], r = road[2];
                if (opened[i] > 0 && opened[j] > 0) {
                    d[i, j] = d[j, i] = Math.Min(d[i, j], r);
                }
            }

            // Floyd-Warshall algorithm
            for (int k = 0; k < n; k++) {
                if (opened[k] > 0) {
                    for (int i = 0; i < n; i++) {
                        if (opened[i] > 0) {
                            for (int j = i + 1; j < n; j++) {
                                if (opened[j] > 0) {
                                    d[i, j] = d[j, i] = Math.Min(d[i, j], d[i, k] + d[k, j]);
                                }
                            }
                        }
                    }
                }
            }

            // Validate
            int good = 1;
            for (int i = 0; i < n; i++) {
                if (opened[i] > 0) {
                    for (int j = i + 1; j < n; j++) {
                        if (opened[j] > 0) {
                            if (d[i, j] > maxDistance) {
                                good = 0;
                                break;
                            }
                        }
                    }
                    if (good == 0) {
                        break;
                    }
                }
            }
            res += good;
        }
        return res;
    }
    
    
    //3112、访问消失节点的最少时间
    public int[] MinimumTime(int n, int[][] edges, int[] disappear) {
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
        int pre=0,nl=possible.Length,tot=possible.Sum();
        for(int i=0;i<nl-1;++i){
            pre+=possible[i];
            if(pre*2-i-1>tot*2-pre*2-nl+i+1) return i+1;
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
                ans = Math.Min(ans, Math.Abs(p1[p][0] - p2[j][0]) + Math.Abs(p1[p][1] - p2[j][1]) + DFS(p + 1, msk ^ (1 << j), p1, p2));
            }
        }

        cache[(p, msk)] = ans;
        return ans;
    }
    
    
    //1186、删除一次得到子数组最大和
    public int MaximumSum(int[] arr) {
        int dp0 = arr[0], dp1 = 0, res = arr[0];
        
        for (int i = 1; i < arr.Length; i++) {
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
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j && IsConnected(bombs, i, j)) {
                    edges.TryAdd(i, new List<int>());
                    edges[i].Add(j);
                }
            }
        }
        
        int res = 0;   // 最多引爆数量
        
        for (int i = 0; i < n; ++i) {
            // 遍历每个炸弹，广度优先搜索计算该炸弹可引爆的数量，并维护最大值
            bool[] visited = new bool[n];
            int cnt = 1;
            Queue<int> queue = new Queue<int>();
            queue.Enqueue(i);
            visited[i] = true;
            while (queue.Count > 0) {
                int cidx = queue.Dequeue();
                foreach (int nidx in edges.ContainsKey(cidx) ? edges[cidx] : new List<int>()) {
                    if (visited[nidx]) {
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
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                set.Add(nums[i] - nums[j]);
            }
        }
        set.Add(INF);
        IList<int> vals = new List<int>(set);
        ((List<int>) vals).Sort();

        int[][][] d = new int[n][][];
        
        for (int i = 0; i < n; i++) {
            d[i] = new int[k + 1][];
            for (int j = 0; j <= k; j++) {
                d[i][j] = new int[vals.Count];
            }
        }
        int[][] border = new int[n][];
        
        for (int i = 0; i < n; i++) {
            border[i] = new int[k + 1];
        }
        int[][] sum = new int[k + 1][];
        
        for (int i = 0; i <= k; i++) {
            sum[i] = new int[vals.Count];
        }
        int[][] suf = new int[n][];
        
        for (int i = 0; i < n; i++) {
            suf[i] = new int[k + 1];
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int pos = BinarySearch(vals, nums[i] - nums[j]);
                
                for (int p = 1; p <= k; p++) {
                    while (border[j][p] < pos) {
                        sum[p][border[j][p]] = (sum[p][border[j][p]] - suf[j][p] + MOD) % MOD;
                        sum[p][border[j][p]] = (sum[p][border[j][p]] + d[j][p][border[j][p]]) % MOD;
                        suf[j][p] = (suf[j][p] - d[j][p][border[j][p]] + MOD) % MOD;
                        border[j][p]++;
                        sum[p][border[j][p]] = (sum[p][border[j][p]] + suf[j][p]);
                    }
                }
            }

            d[i][1][vals.Count - 1] = 1;
            
            for (int p = 2; p <= k; p++) {
                for (int v = 0; v < vals.Count; v++) {
                    d[i][p][v] = sum[p - 1][v];
                }
            }
            
            for (int p = 1; p <= k; p++) {
                for (int v = 0; v < vals.Count; v++) {
                    suf[i][p] = (suf[i][p] + d[i][p][v]) % MOD;
                }
                sum[p][0] = (sum[p][0] + suf[i][p]) % MOD;
            }
        }

        int res = 0;
        
        for (int i = 0; i < n; i++) {
            for (int v = 0; v < vals.Count; v++) {
                res = (int) ((res + 1L * vals[v] * d[i][k][v] % MOD) % MOD);
            }
        }
        
        return res;
    }
    
    int BinarySearch(IList<int> vals, int target) {
        int low = 0, high = vals.Count;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (vals[mid] >= target) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }
    
    
    //2766、重新放置石块
    public IList<int> RelocateMarbles(int[] nums, int[] moveFrom, int[] moveTo) {
        var theSet = nums.ToHashSet();
        for (int i = 0; i < moveFrom.Length; i++) {
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
            if(num[i]=='0'&&!has0)
            {
                has0 = true;
            }
            else if(num[i]=='0'&&has0)
            {
                return nl - i - 2;
            }
            else if(num[i]=='5'&&has0)
            {
                return nl - i - 2;
            }
            else if(num[i]=='5'&&!has0)
            {
                has5 = true;
            }
            else if(num[i] == '2'&&has5)
            {
                return nl - i - 2;
            }
            else if(num[i]=='7'&&has5)
            {
                return nl - i - 2;
            }
        }

        return has0 ? nl - 1 : nl;
    }

}
