namespace ConsoleAppTest;

public class TestPro
{
    static void Main()
    {
        // int[] nums = new int[] { 2, 7, 11, 15 };
        // int[] nums1 = new int[] { 4, 3, 2, 1 };
        // int[] nums2 = new[] { 2, 0, -2, -5, -5, -3, 2, -4 };
        // int[] nums3 = new[] { 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1 };
        // int[] nums4 = new[] { 4, 2, 0, 3, 2, 5 };
        // int[] nums = new[] { 9, 6, 8, 8, 5, 6, 3 };
        // int[] nums = new[] { 7, 9, 8, 5, 0, 0, 4, 2, 7, 6, 0, 8, 1, 2, 3 };
        // int[] nums1 = new[] { 2, 0 };
        // int[] nums2 = new[] { 1 };
        
        
        
        // int m = 8;
        // int n = 1;
        // int target = 121;

        // string s = "aaabb";
        // string s = "cbaebabacd";
        // string s = "abcabcbb";
        // string str1 = "   fly me   to   the moon  ";
        // string p = "bb";
        // string num = "2245047";


        ListNode node = new ListNode(1, null);

        // Daily solution = new Daily();
        // Program180Ti solution = new Program180Ti();
        // Solution solution = new Solution();
        Program100Ti solution = new Program100Ti();
        
        // var x = solution.Merge(nums1, m, nums2, n);

        // solution.Merge(nums1, m, nums2, n);
            
        Console.WriteLine(solution.SwapPairs(node).val);
        
        // foreach (var VARIABLE in x)
        // {
        //     foreach (var i in VARIABLE)
        //     {
        //         Console.WriteLine(i);
        //         
        //     }
        // }
        
        // foreach (var i in nums1)
        // {
        //     Console.WriteLine(i);
        //     
        // }
        
    }
}