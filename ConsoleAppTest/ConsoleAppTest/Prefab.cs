namespace ConsoleAppTest;

public static class Prefab
{
        public static int Recursive1(int n)
        {
            int x = 0, ans = 0;
            if (n == 1)
                return 0;
            for (int i = 1; i < n; i++)
            {
                ans = x + i;
                x = ans;
            }
            return ans;
        }
        
        
        public static int Recursive2(int n)
        {
            return (n * n - n) / 2;
        }
}

public class ListNode
{
    public int val;
    public ListNode next;

    public ListNode(int val = 0, ListNode next = null)
    {
        this.val = val;
        this.next = next;
    }
}

public class TreeNode
{
    public int val;
    public TreeNode left;
    public TreeNode right;

    public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
    {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}