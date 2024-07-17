using System.Runtime.InteropServices.Marshalling;

namespace ConsoleAppTest;

public class Program180Ti
{
    //88、合并两个有序数组
    public void MergeTwoArray(int[] nums1, int m, int[] nums2, int n)
    {
        if(n == 0)
            return;
        if (m == 0)
        {
            for (int i = 0; i < n; i++)
            {
                nums1[i] = nums2[i];
            }
        }
        else
        {
            int x1 = 0, x2 = 0;
            int[] res = new int[m];
            Array.Copy(nums1, res, m);
            for (int i = 0; i < m+n; i++)
            {
                if (x1 < m && x2 < n)
                {
                    if (res[x1] <= nums2[x2])
                    {
                        nums1[i] = res[x1];
                        x1++;
                    }
                    else
                    {
                        nums1[i] = nums2[x2];
                        x2++;
                    }
                }else if (x1 < m)
                {
                    nums1[i] = res[x1];
                    x1++;
                }
                else
                {
                    nums1[i] = nums2[x2];
                    x2++;
                }
            }
        }
    }
    
    
    //27、移除元素
    public int RemoveElement(int[] nums, int val)
    {
        var nl = nums.Length;

        // var i = 0;
        // int j = n - 1;
        // while (i <= j) {
        //     if (nums[i] == val) {
        //         nums[i] = nums[j];
        //         j--;
        //     } else {
        //         i++;
        //     }
        // }
        // return i;

        Array.Sort(nums);
        int n = 0;
        int removeBegin = 0;
        int removeEnd = 0;
        bool removecheck = false;
        for (int i = 0; i < nl; i++)
        {
            if (nums[i] == val)
            {
                n++;
                if (!removecheck)
                {
                    removeBegin = i;
                    removecheck = true;
                }
                else
                {
                    continue;
                }
            }
            else
            {
                if (removecheck)
                {
                    removeEnd = i;
                    removecheck = false;
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            nums[removeBegin + i] = nums[nl - i - 1];
        }

        return nums.Length - n;
    }
    
    
    //26、删除有序数组中的重复项
    public int RemoveDuplicates(int[] nums)
    {
        List<int> listnums = new List<int>();
        int n = 0;
        foreach (var i in nums)
        {
            if (listnums.Contains(i))
            {
            }
            else
            {
                listnums.Add(i);
                n++;
            }
        }

        for (int i = 0; i < n; i++)
        {
            nums[i] = listnums[i];
        }

        return n;
    }
    
    
    //80、删除有序数组中的重复项2
    public int RemoveDuplicates2(int[] nums)
    {
        Dictionary<int, int> dicNums = new Dictionary<int, int>();
        int n = 0;
        List<int> listNums = new List<int>();
        
        foreach (var i in nums)
        {
            if (!dicNums.ContainsKey(i))
            {
                dicNums.Add(i, 1);
                listNums.Add(i);
                n++;
            }
            else
            {
                if (dicNums[i] == 2)
                {
                }
                else
                {
                    dicNums[i]++;
                    listNums.Add(i);
                    n++;
                }
                    
            }
        }

        for (int i = 0; i < n; i++)
        {
            nums[i] = listNums[i];
        }
        
        return n;
    }
    
    
    //169、多数元素
    public int MajorityElement(int[] nums)
    {
        Dictionary<int, int> numCount = new Dictionary<int, int>();
        int ans = nums[0], count = 0;
        foreach (var i in nums)
        {
            if (!numCount.TryAdd(i, 1))
            {
                numCount[i]++;
            }

            if (count < numCount[i])
            {
                ans = i;
                count = numCount[i];
            }
        }

        return ans;
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
    
    
    //121、买卖股票的最佳实机
    public int MaxProfit1(int[] prices)
    {
        int l = 0, r = 1, ans = 0;
        while (r < prices.Length)
        {
            ans = Math.Max(prices[r] - prices[l], ans);
            if (prices[l] > prices[r])
            {
                l++;
                continue;
            }

            r++;
        }

        return ans;
    }
    
    
    //122、买卖股票的最佳时机2
    public int MaxProfit2(int[] prices)
    {
        return 0;
    }
}