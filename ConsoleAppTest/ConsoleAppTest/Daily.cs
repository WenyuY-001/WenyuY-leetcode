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
}