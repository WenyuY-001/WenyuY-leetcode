// See https://aka.ms/new-console-template for more information

namespace ConsoleAppTest;

using System.Collections;

public class Solution
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
            if (hash.ContainsKey(x))
            {
                return new[] { (int)hash[x], i };
            }

            if (hash.ContainsKey(nums[i]))
                continue;
            hash.Add(nums[i], i);
        }

        return new int[2];
    }


    //9、回文数
    public bool IsPalindrome(int x)
    {
        if (x < 0 || x % 10 == 0)
            return false;
        if (x < 10)
            return true;
        int revertnum = 0;
        while (x > revertnum)
        {
            revertnum = revertnum * 10 + x % 10;
            x /= 10;
        }

        return x == revertnum || x == revertnum / 10;
    }


    //13、罗马数字转整数
    public int RomanToInt(string s)
    {
        Dictionary<char, int> symbolValue = new Dictionary<char, int>()
        {
            { 'I', 1 }, { 'V', 5 }, { 'X', 10 }, { 'L', 50 }, { 'C', 100 }, { 'D', 500 }, { 'M', 1000 }
        };
        int ans = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (i < s.Length - 1 && symbolValue[s[i]] < symbolValue[s[i + 1]])
            {
                ans -= symbolValue[s[i]];
            }
            else
            {
                ans += symbolValue[s[i]];
            }
        }

        return ans;
    }


    //14、最长公共前缀
    public string LongestCommonPrefix(string[] strs)
    {
        string ans = "";
        char charCheck = ' ';
        int n1 = strs.Min().Length;
        int n2 = strs.Length;
        if (n2 == 1)
            return strs[0];
        for (int j = 0; j < n1; j++)
        {
            for (int i = 0; i < n2; i++)
            {
                if (i == 0)
                    charCheck = strs[i][j];
                else
                {
                    if (charCheck == strs[i][j])
                    {
                        if (i == n2 - 1)
                            ans += charCheck;
                        continue;
                    }
                    else
                    {
                        goto end;
                    }
                }
            }
        }

        end:
        return ans;
    }


    //20、有效的括号
    public bool IsValid(string s)
    {
        var stack = new Stack<char>();
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '(')
                stack.Push(')');
            else if (s[i] == '[')
                stack.Push(']');
            else if (s[i] == '{')
                stack.Push('}');

            else if (stack.Any() || stack.Pop() != s[i])
                return false;
        }

        if (stack.Any())
            return false;
        else
            return true;
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

        return answer.next;
    }


    //26、删除有序数组中的重复项
    public int RemoveDuplicates(int[] nums)
    {
        Dictionary<int, int> dicnums = new Dictionary<int, int>();
        List<int> listnums = new List<int>();
        int n = 0;
        foreach (var i in nums)
        {
            if (listnums.Contains(i))
            {
                continue;
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


    //28、找出字符串中第一个匹配项的下标
    public int StrStr(string haystack, string needle)
    {
        return haystack.IndexOf(needle);
    }


    //35、搜索插入位置
    public int SearchInsert(int[] nums, int target)
    {
        int n = nums.Length;
        int left = 0, right = n - 1, ans = n;
        if (nums[0] == target)
            return 0;
        if (nums[right] == target)
            return right;
        while (left <= right)
        {
            int mid = ((right - left) >> 1) + left;
            if (target <= nums[mid])
            {
                ans = mid;
                right = mid - 1;
            }
            else
            {
                left = mid + 1;
            }
        }

        return ans;
    }


    //58、最后一个单词的长度
    public int LengthOfLastWord(string s)
    {
        string[] sps = s.Split(' ');
        int strl = sps.Length;
        for (int i = strl - 1; i >= 0; i--)
        {
            if (sps[i] != "")
            {
                return sps[i].Length;
            }
        }

        return 0;
    }


    //66、加一
    public int[] PlusOne(int[] digits)
    {
        int nl = digits.Length;
        int[] digit = new int[nl + 1];
        digits[nl - 1]++;
        for (int i = nl - 1; i >= 0; i--)
        {
            if (!(digits[i] < 10))
            {
                if (i == 0)
                {
                    digit[1] = 0;
                    digit[i] = 1;
                    break;
                }

                digits[i - 1]++;
                digit[i + 1] = 0;
                digit[i] = digits[i - 1];
                digits[i] = 0;
            }
            else
            {
                break;
            }
        }

        if (digit[0] != 0)
        {
            digits = digit;
        }

        return digits;
    }


    //67、二进制求和（鸽了）
    public string AddBinary(string a, string b)
    {
        return a + b;
    }


    //69、x的平方根（二分查找）（鸽了）
    public int MySqrt(int x)
    {
        return (int)Math.Pow(x, 0.5);
    }


    //70、爬楼梯
    public int ClimbStairs(int n)
    {
        int[] nums = new int[49];
        nums[0] = 1;
        nums[1] = 2;
        if (n < 3)
        {
            return nums[n - 1];
        }

        for (int i = 2; i < n; i++)
        {
            nums[i] = nums[i - 1] + nums[i - 2];
        }

        return nums[n - 1];
    }


    //83、删除排序链表中的重复元素
    public ListNode DeleteDuplicates(ListNode head)
    {
        if (head == null)
        {
            return null;
        }

        ListNode ans = head;
        var temp = head.next;
        while (temp != null)
        {
            if (head.val == temp.val)
            {
                head.next = temp.next;
                temp = temp.next;
            }
            else
            {
                head = head.next;
                temp = temp.next;
            }
        }

        return ans;
    }
    
    
    //88、合并两个有序数组（有问题）
    public void Merge(int[] nums1, int m, int[] nums2, int n) {
        if(n == 0)
            return;
        else if (m == 0)
        {
            for (int i = 0; i < n; i++)
            {
                nums1[i] = nums2[i];
            }
            return;
        }
        else
        {
            int x1 = 0, x2 = 0;
            int[] res = nums1;
            for (int i = 0; i < m+n; i++)
            {
                if (res[x1] <= nums2[x2] && x1 < m)
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
    
    
    //94、二叉树的中序遍历（鸽了）
    public IList<int> InorderTraversal(TreeNode root)
    {
        return null;
    }
    
    
    //100、相同的树（鸽了）
    public bool IsSameTree(TreeNode p, TreeNode q)
    {
        return false;
    }
}