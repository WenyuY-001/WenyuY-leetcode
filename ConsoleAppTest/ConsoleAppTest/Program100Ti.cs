using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices.ComTypes;
using System.Text.Json.Serialization.Metadata;

namespace ConsoleAppTest;

public class Program100Ti
{
    //1、两数之和
    public int[] TwoSum(int[] nums, int target)
    {
        int x;
        Dictionary<int, int> hash = new Dictionary<int, int>();
        for (int i = 0; i < nums.Length; i++)
        {
            x = 0;
            x = target - nums[i];

            if (hash.TryGetValue(x, out var value))
            {
                return new[] { value, i };
            }

            if (hash.ContainsKey(nums[i]))
                continue;
            hash.Add(nums[i], i);
        }

        return new int[2];
    }


    //49、字母异位词分组
    public IList<IList<string>> GroupAnagrams(string[] strs)
    {
        IList<IList<string>> ans = new List<IList<string>>();
        IDictionary<string, int> teChars = new Dictionary<string, int>(); //第二位可以设置为string列表
        int i = 0;
        foreach (var str in strs)
        {
            char[] sChars = str.ToCharArray();
            Array.Sort(sChars);
            string str1 = new string(sChars);

            if (!teChars.ContainsKey(str1))
            {
                Console.WriteLine("in");
                teChars.Add(str1, i);
                ans.Add(new List<string>() { str });
                i++;
            }
            else
            {
                ans[teChars[str1]].Add(str);
            }
        }

        return ans;
    }


    //128、最长连续序列
    public int LongestConsecutive(int[] nums)
    {
        if (nums.Length == 0)
            return 0;
        Array.Sort(nums);
        int length = 0;
        int ans = 0;
        int valCheck = nums[0];
        foreach (var num in nums)
        {
            if (num == valCheck)
            {
                length++;
                valCheck++;
            }
            else if (num < valCheck)
                continue;
            else
            {
                ans = Math.Max(ans, length);
                valCheck = num + 1;
                length = 1;
            }
        }

        ans = Math.Max(ans, length);

        return ans;
    }


    //283、移动零
    public void MoveZeroes(int[] nums)
    {
        int nl = nums.Length;
        int left = 0, right = 1;
        while (left < right)
        {
            if (right == nl)
                break;

            if (nums[left] != 0)
            {
                left++;
            }

            if (nums[left] == 0 && nums[right] != 0)
            {
                nums[left] = nums[right];
                nums[right] = 0;
            }

            right++;
        }
    }


    //11、盛最多水的容器
    public int MaxArea(int[] height)
    {
        int nl = height.Length;
        int ans = 0, left = 0, right = nl - 1;
        int nowArea = 0;
        while (left < right)
        {
            nowArea = Math.Min(height[left], height[right]) * (right - left);
            if (height[left] < height[right])
            {
                left++;
            }
            else
            {
                right--;
            }

            ans = Math.Max(ans, nowArea);
        }

        return ans;
    }


    //15、三数之和（双指针）
    public IList<IList<int>> ThreeSum(int[] nums)
    {
        IList<IList<int>> ans = new List<IList<int>>();
        Array.Sort(nums);
        int nl = nums.Length;
        if (nl == 3)
        {
            if (nums.Sum() == 0)
                ans.Add(nums.ToList());
            return ans;
        }

        if (nl < 3)
            return ans;
        if (nums[0] == 0 && nums[^1] == 0)
        {
            ans.Add(new List<int>() { 0, 0, 0 });
            return ans;
        }

        int left = 0, right = 0;
        for (int i = 0; i < nl - 2; i++)
        {
            if (i != 0 && nums[i] == nums[i - 1])
                continue;
            if (nums[i] > 0) break;
            left = i + 1;
            right = nl - 1;
            while (left < right)
            {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum < 0)
                    left++;
                else if (sum > 0)
                    right--;
                else
                {
                    ans.Add(new List<int>() { nums[i], nums[left], nums[right] });
                    left++;
                    right--;
                    while (left < nl - 1 && nums[left] == nums[left - 1]) left++;
                    while (nums[right] == nums[right + 1]) right--;
                }
            }
        }

        return ans;
    }


    //42、接雨水（速度很慢，最好双指针）
    public int Trap(int[] height)
    {
        int ans = 0;
        int nl = height.Length;
        int left = 0, right = 1, count = 0, maxRight = 0, maxPos = nl - 1;
        while (left < right && right < nl && left < nl - 1)
        {
            if (height[left] == 0)
            {
                left++;
                right++;
            }
            else if (height[left] > height[right])
            {
                if (height[right] >= maxRight)
                {
                    maxRight = height[right];
                    maxPos = right;
                }

                right++;
                count++;
            }
            else if (height[right] >= height[left])
            {
                if (count == 0)
                {
                    left++;
                    right++;
                    continue;
                }

                var xzHeight = height[left];
                while (left < right - 1)
                {
                    left++;
                    ans += xzHeight - height[left];
                }

                maxRight = 0;
                count = 0;
                left++;
                right++;
            }

            if (right == nl && count != 0)
            {
                var xzHeight = maxRight;
                right = maxPos;
                if (left == right - 1)
                {
                    left++;
                    right++;
                    maxRight = 0;
                }
                else
                {
                    while (left < right - 1)
                    {
                        left++;
                        ans += xzHeight - height[left];
                    }

                    maxRight = 0;
                }
            }
        }

        return ans;
    }


    //3、无重复字符的最长字串
    public int LengthOfLongestSubstring(string s)
    {
        List<char> list = new List<char>();
        int left = 0, right = 0;
        int nowLength = 0, ans = 0;
        while (left < s.Length && right < s.Length)
        {
            if (list.Contains(s[right]))
            {
                list.Remove(s[left]);
                left++;
                nowLength--;
            }
            else
            {
                list.Add(s[right]);
                right++;
                nowLength++;
            }

            ans = ans < nowLength ? nowLength : ans;
        }

        return ans;
    }


    //438、找到字符串中所有字母异位词
    public IList<int> FindAnagrams(string s, string p)
    {
        IList<int> ans = new List<int>();
        if (s.Length < p.Length) return ans;

        int[] pCount = new int[26];
        int[] sCount = new int[26];

        // 初始化计数数组
        foreach (char c in p)
        {
            pCount[c - 'a']++;
        }

        int left = 0, right = 0;
        // 初始化滑动窗口
        while (right < p.Length)
        {
            sCount[s[right] - 'a']++;
            right++;
        }

        right--;

        // 滑动窗口遍历字符串 s
        while (right < s.Length)
        {
            if (Enumerable.SequenceEqual(pCount, sCount))
            {
                ans.Add(left);
            }

            right++;
            if (right != s.Length)
            {
                sCount[s[right] - 'a']++;
            }

            sCount[s[left] - 'a']--;
            left++;
        }

        return ans;


        // IList<int> ans = new List<int>();
        // char[] ps = p.ToCharArray();
        // Array.Sort(ps);
        // int left = 0, right = p.Length;
        // while (left < s.Length - right + 1)
        // {
        //     char[] nowS = s.Substring(left, right).ToCharArray();
        //     Array.Sort(nowS);
        //     if (nowS.SequenceEqual(ps))
        //         ans.Add(left);
        //     left++;
        // }
        // return ans;
    }


    //560、和为K的子数组
    public int SubarraySum(int[] nums, int k)
    {
        Array.Sort(nums);
        var dict = new Dictionary<int, int> { { 0, 1 } };
        int sum = 0, res = 0;
        for (int i = nums.Length - 1; i >= 0; i--)
        {
            sum += nums[i];
            if (dict.TryGetValue(sum - k, out var len)) res += len;
            if (dict.ContainsKey(sum)) dict[sum]++;
            else dict.Add(sum, 1);
        }

        return res;
    }


    //239、滑动窗口最大值
    public int[] MaxSlidingWindow(int[] nums, int k)
    {
        if (nums.Length < k + 1)
            return new[] { nums.Max() };

        int[] ans = new int[nums.Length - k + 1];
        PriorityQueue<int, int> priorityQueue = new PriorityQueue<int, int>();
        for (int i = 0; i < k - 1; i++)
        {
            priorityQueue.Enqueue(i, -nums[i]);
        }

        for (int i = k - 1; i < nums.Length; i++)
        {
            int ns = i - k + 1;
            priorityQueue.Enqueue(i, -nums[i]);
            if (priorityQueue.Count > k)
            {
                while (priorityQueue.Peek() < ns)
                {
                    priorityQueue.Dequeue();
                }
            }

            ans[ns] = nums[priorityQueue.Peek()];
        }

        return ans;
    }


    //76、最小覆盖子串（鸽了）
    public string MinWindow(string s, string t)
    {
        Dictionary<string, int> ans = new Dictionary<string, int>();
        Dictionary<char, int> tc = new Dictionary<char, int>();
        foreach (var i in t)
        {
            if (!tc.ContainsKey(i))
                tc.Add(i, 1);
            else
                tc[i]++;
        }

        int sl = s.Length, tl = t.Length;
        int left = 0, right = 0;
        Dictionary<char, int> sc = tc;
        while (left < sl - tl)
        {
            if (sc.ContainsKey(s[left]))
            {
                sc[s[left]]--;
                right = left + 1;
                while (right > left)
                {
                    if (sc.ContainsKey(s[right]))
                    {
                        sc[s[right]]--;
                        if (sc.SequenceEqual(tc))
                        {
                            string anss = s.Substring(left, right - left + 1);
                            ans.Add(anss, anss.Length);
                            break;
                        }
                    }

                    right--;
                }
            }

            left++;
        }

        return null;
    }


    //53、最大子数组和
    public int MaxSubArray(int[] nums)
    {
        int nl = nums.Length;
        int ans = Int32.MinValue;
        int[] sums = new int[nl];
        sums[0] = nums[0];
        for (int i = 1; i < nums.Length; i++)
        {
            sums[i] = Math.Max(sums[i - 1] + nums[i], nums[i]);
            ans = sums[i] > ans ? sums[i] : ans;
        }

        ans = sums[0] > ans ? sums[0] : ans;
        return ans;
    }


    //56、合并区间
    public int[][] Merge(int[][] intervals)
    {
        if (intervals.Length == 0)
        {
            return intervals;
        }

        intervals = intervals.OrderBy(p => p[0]).ToArray();
        List<int[]> list = new List<int[]>();
        for (int i = 0; i < intervals.Length - 1; i++)
        {
            if (intervals[i][1] >= intervals[i + 1][0])
            {
                intervals[i + 1][0] = intervals[i][0];

                if (intervals[i][1] >= intervals[i + 1][1])
                {
                    intervals[i + 1][1] = intervals[i][1];
                }
            }
            else
            {
                list.Add(intervals[i]);
            }
        }

        list.Add(intervals[^1]);

        return list.ToArray();
    }


    //189、轮转数组
    public void Rotate(int[] nums, int k)
    {
        int nl = nums.Length;
        k = k % nl;
        int[] res = new int[nl];
        Array.Copy(nums, 0, res, k, nl - k);
        Array.Copy(nums, nl - k, res, 0, k);
        Array.Copy(res, nums, nl);
    }


    //238、除自身以外数组的乘积
    public int[] ProductExceptSelf(int[] nums)
    {
        int n = nums.Length;
        int pre = 1, suf = 1;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++)
        {
            ans[i] = pre;
            pre *= nums[i];
        }

        for (int j = n - 1; j >= 0; j--)
        {
            ans[j] *= suf;
            suf *= nums[j];
        }

        return ans;
    }


    //41、缺失的第一个正数
    public int FirstMissingPositive(int[] nums)
    {
        Array.Sort(nums);
        int ans = 1;
        foreach (var i in nums)
        {
            if (i == ans)
            {
                ans++;
            }
        }

        return ans;
    }


    //73、矩阵置零
    public void SetZeroes(int[][] matrix)
    {
        int cl = matrix[0].Length, ll = matrix.Length;
        bool c0 = false, l0 = false;
        for (int i = 0; i < cl; i++)
        {
            if (matrix[0][i] == 0)
            {
                c0 = true;
                break;
            }
        }

        for (int i = 0; i < ll; i++)
        {
            if (matrix[i][0] == 0)
            {
                l0 = true;
                break;
            }
        }

        for (int i = 1; i < ll; i++)
        {
            for (int j = 1; j < cl; j++)
            {
                if (matrix[i][j] == 0)
                {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }

        for (int i = 1; i < cl; i++)
        {
            if (matrix[0][i] == 0)
            {
                for (int j = 1; j < ll; j++)
                {
                    matrix[j][i] = 0;
                }
            }
        }

        for (int i = 1; i < ll; i++)
        {
            if (matrix[i][0] == 0)
            {
                for (int j = 1; j < cl; j++)
                {
                    matrix[i][j] = 0;
                }
            }
        }

        if (c0)
        {
            for (int i = 0; i < cl; i++)
            {
                matrix[0][i] = 0;
            }
        }

        if (l0)
        {
            for (int i = 0; i < ll; i++)
            {
                matrix[i][0] = 0;
            }
        }
    }


    //54、螺旋矩阵
    public IList<int> SpiralOrder(int[][] matrix)
    {
        IList<int> ans = new List<int>();
        int m = matrix.Length, n = matrix[0].Length;
        int count = m * n;
        int top = 0, bottom = m - 1, left = 0, right = n - 1;
        while (count > 0)
        {
            for (int col = left; col <= right && count > 0; col++)
            {
                ans.Add(matrix[top][col]);
                count--;
            }

            top++;
            for (int row = top; row <= bottom && count > 0; row++)
            {
                ans.Add(matrix[row][right]);
                count--;
            }

            right--;
            for (int col = right; col >= left && count > 0; col--)
            {
                ans.Add(matrix[bottom][col]);
                count--;
            }

            bottom--;
            for (int row = bottom; row >= top && count > 0; row--)
            {
                ans.Add(matrix[row][left]);
                count--;
            }

            left++;
        }

        return ans;
    }


    //48、旋转图像
    public void Rotate(int[][] matrix)
    {
        int nl = matrix.Length;
        //首先进行上下翻转
        for (int i = 0; i < nl / 2; i++)
        {
            (matrix[i], matrix[nl - i - 1]) = (matrix[nl - i - 1], matrix[i]);
        }

        //然后进行对角线翻转
        for (int i = 0; i < nl; i++)
        {
            for (int j = i; j < nl; j++)
            {
                (matrix[i][j], matrix[j][i]) = (matrix[j][i], matrix[i][j]);
            }
        }
    }
    
    
    //240、搜索二维矩阵2
    public bool SearchMatrix2(int[][] matrix, int target)
    {
        int m = matrix.Length, n = matrix[0].Length;
        int x = 0, y = n - 1;
        while (x < m && y >= 0) {
            if (matrix[x][y] == target) {
                return true;
            }
            if (matrix[x][y] > target) {
                --y;
            } else {
                ++x;
            }
        }
        return false;
    }
    
    
    //160、相交链表
    public ListNode GetIntersectionNode(ListNode headA, ListNode headB) {
        if (headA==null || headB==null) 
        {
            return null;
        }
        ListNode you = headA, she = headB;
        while (you != she) 
        { 
            you = you != null ? you.next : headB; 
            she = she != null ? she.next : headA; 
        }
        
        return you;
    }
    
    
    //206、反转链表
    public ListNode ReverseList(ListNode head) {
        if(head == null){
            return head;
        }
        ListNode l1 = null;
        ListNode l2 = head;
        while(l2 != null){
            ListNode l3 = l2.next;
            l2.next = l1;
            l1 = l2;
            l2 = l3; 
        }
        return l1;
    }
    
    
    //234、回文链表
    public bool IsPalindrome(ListNode head)
    {
        ListNode slow = head, fast = head,  prev = null;
        while (fast != null){//find mid node
            slow = slow.next;
            fast = fast.next != null ? fast.next.next: fast.next;
        }
        while (slow != null){//reverse
            ListNode temp = slow.next;
            slow.next = prev;
            prev = slow;
            slow = temp;
        }
        while (head != null && prev != null){//check
            if (head.val != prev.val){
                return false;
            }
            head = head.next;
            prev = prev.next;
        }
        return true;
    }
    
    
    //141、环状链表
    public bool HasCycle(ListNode head) {
        if (head == null || head.next == null)
        {
            return false;
        }
        var slow = head;
        var fast = head.next;
        
        while(fast != null)
        {
            if(fast.next == null) 
                return false;
            if(slow == fast) 
                return true;
            fast = fast.next.next;
            slow = slow.next;
        }
        
        return false;
    }
    
    
    //142、环状链表2
    public ListNode DetectCycle(ListNode head) {
        if (head == null || head.next == null)
        {
            return null;
        }
        var slow = head;
        var fast = head;
        
        while(fast != null)
        {
            if(fast.next == null) 
                return null;
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast)
            {
                ListNode x = head, y = slow;
                while (x!=y)
                {
                    x = x.next;
                    y = y.next;
                }
                return x;
            }
        }
        
        return null;
    }
    
    
    //21、合并两个有序链表
    public ListNode MergeTwoLists(ListNode list1, ListNode list2)
    {
        if (list1 == null) return list2;
        if (list2 == null) return list1;

        ListNode answer = new ListNode(0);
        ListNode ans = answer;

        if (list1.val <= list2.val)
        {
            ans.val = list1.val;
            list1 = list1.next;
        }
        else
        {
            ans.val = list2.val;
            list2 = list2.next;
        }

        while (list1 != null && list2 != null)
        {
            if (list1.val <= list2.val)
            {
                ans.next = list1;
                list1 = list1.next;
            }
            else
            {
                ans.next = list2;
                list2 = list2.next;
            }

            ans = ans.next;
        }

        ans.next = list1 == null ? list2 : list1;

        return answer;
    }
    
    
    //2、两数相加
    public ListNode AddTwoNumbers(ListNode l1, ListNode l2)
    {
        ListNode head = null, tail = null;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int n1 = l1?.val ?? 0;
            int n2 = l2?.val ?? 0;
            int sum = n1 + n2 + carry;
            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }
            carry = sum / 10;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry > 0) {
            tail.next = new ListNode(carry);
        }
        return head;
    }
    
    
    //19、删除链表的倒数第N个节点
    public ListNode RemoveNthFromEnd(ListNode head, int n)
    {
        ListNode fHead = new ListNode(0);
        fHead.next = head;
        ListNode fast = fHead;
        ListNode slow = fHead;
        
        for (int i = 0; i <= n; i++)
        {
            fast = fast.next;
        }
        
        while (fast != null)
        {
            fast = fast.next;
            slow = slow.next;
        }
        
        slow.next = slow.next.next;

        return fHead.next;
    }
    
    
    //24、两两交换链表中的节点
    public ListNode SwapPairs(ListNode head)
    {
        ListNode cur = head;
        ListNode sec = head.next != null ? head.next : null;
        bool isCh = false;
        while (cur != null && cur.next != null)
        {
            if (!isCh)
            {
                isCh = true;
            }
            (cur.next, cur.next.next) = (cur.next.next, cur);
            cur = cur.next;
        }

        return isCh ? sec : head;
    }
    
    
    //34、在排序数组中查找元素的第一个和最后一个位置
    public int[] SearchRange(int[] nums, int target)
    {
        int[] arr = [-1, -1];
        int left = 0, right = nums.Length - 1;
        bool lad = true, rad = true;

        while (left < right)
        {
            if (nums[left] == target)
            {
                arr[0] = left;
                lad = false;
            }

            if (nums[right] == target)
            {
                arr[1] = right;
                rad = false;
            }

            if (lad)
                left++;
            if (rad)
                right++;
            if (!rad && !lad)
                break;
        }
        for (int i = 0; i < nums.Length ; i++)
        {
            if (nums[i] == target)
            {
                arr[0] = i;
                break;
            }
        }

        if (arr[0] is -1)
            return arr;
            
        for (int j = nums.Length - 1; j >= 0; j--)
        {
            if (nums[j] == target)
            {
                arr[1] = j;
                break;
            }
        }
        
        return arr;
    }
}