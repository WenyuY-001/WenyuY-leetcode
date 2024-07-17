﻿using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices.ComTypes;
using System.Text.Json.Serialization.Metadata;

namespace ConsoleAppTest;

public class Program100Ti
{
    //1、两数之和
    public int[] TwoSum(int[] nums, int target) {
        int x;
        Dictionary<int, int> hash = new Dictionary<int, int>();
        for (int i = 0; i < nums.Length; i++)
        {
            x = 0;
            x = target - nums[i];
            
            if (hash.TryGetValue(x, out var value))
            {
                return new []{value, i};
            }
            if(hash.ContainsKey(nums[i]))
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
                teChars.Add(str1,i);
                ans.Add(new List<string>(){str});
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
                valCheck = num+1;
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
        while (left<right)
        {
            if (right == nl)
                break;
            
            if (nums[left]!=0)
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
            if (height[left] <height[right])
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
                    ans.Add(new List<int>(){nums[i],nums[left],nums[right]});
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
        foreach (char c in p) {
            pCount[c - 'a']++;
        }

        int left = 0, right = 0;
        // 初始化滑动窗口
        while (right < p.Length) {
            sCount[s[right] - 'a']++;
            right++;
        }
        right--;

        // 滑动窗口遍历字符串 s
        while (right < s.Length) {
            if (Enumerable.SequenceEqual(pCount, sCount)) {
                ans.Add(left);
            }
            right++;
            if (right != s.Length) {
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
    public int SubarraySum(int[] nums, int k) {
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
        for (int i = 0; i < k-1; i++)
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
            sums[i] = Math.Max(sums[i-1] + nums[i], nums[i]);
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
        Array.Copy(res,nums,nl);
    }
    
    
    //238、除自身以外数组的乘积
    public int[] ProductExceptSelf(int[] nums)
    {
        int n = nums.Length;
        int pre = 1, suf = 1;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            ans[i] = pre;
            pre *= nums[i];
        }
        for (int j = n - 1; j >= 0; j--) {
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
    public IList<int> SpiralOrder(int[][] matrix) {
        
    }
}