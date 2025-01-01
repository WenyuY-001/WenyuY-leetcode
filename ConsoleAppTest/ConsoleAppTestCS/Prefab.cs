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

public class Node {
    public int val;
    public Node next;
    public Node random;
    
    public Node(int _val) {
        val = _val;
        next = null;
        random = null;
    }
}

public class Employee {
    public int id;
    public int importance;
    public IList<int> subordinates;
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

public class UnionFind
{
    private int[] _ancestor;

    public UnionFind(int n) {
        _ancestor = new int[n];
        for (int i = 0; i < n; ++i) {
            _ancestor[i] = i;
        }
    }

    public void Union(int index1, int index2) {
        _ancestor[Find(index1)] = Find(index2);
    }

    public int Find(int index) {
        if (_ancestor[index] != index) {
            _ancestor[index] = Find(_ancestor[index]);
        }
        return _ancestor[index];
    }
}

public class SegNode {
    public long v00, v01, v10, v11;

    public SegNode() {
        v00 = v01 = v10 = v11 = 0;
    }

    public void Set(long v) {
        v00 = v01 = v10 = 0;
        v11 = Math.Max(v, 0);
    }

    public long Best() {
        return v11;
    }
}

public class SegTree {
    private int n;
    private SegNode[] tree;

    public SegTree(int n) {
        this.n = n;
        tree = new SegNode[n * 4 + 1];
        for (int i = 0; i < tree.Length; i++) {
            tree[i] = new SegNode();
        }
    }

    public void Init(int[] nums) {
        InternalInit(nums, 1, 1, n);
    }

    public void Update(int x, int v) {
        InternalUpdate(1, 1, n, x + 1, v);
    }

    public long Query() {
        return tree[1].Best();
    }

    private void InternalInit(int[] nums, int x, int l, int r) {
        if (l == r) {
            tree[x].Set(nums[l - 1]);
            return;
        }
        int mid = (l + r) / 2;
        InternalInit(nums, x * 2, l, mid);
        InternalInit(nums, x * 2 + 1, mid + 1, r);
        Pushup(x);
    }

    private void InternalUpdate(int x, int l, int r, int pos, int v) {
        if (l > pos || r < pos) {
            return;
        }
        if (l == r) {
            tree[x].Set(v);
            return;
        }
        int mid = (l + r) / 2;
        InternalUpdate(x * 2, l, mid, pos, v);
        InternalUpdate(x * 2 + 1, mid + 1, r, pos, v);
        Pushup(x);
    }

    private void Pushup(int x) {
        int l = x * 2, r = x * 2 + 1;
        tree[x].v00 = Math.Max(tree[l].v00 + tree[r].v10, tree[l].v01 + tree[r].v00);
        tree[x].v01 = Math.Max(tree[l].v00 + tree[r].v11, tree[l].v01 + tree[r].v01);
        tree[x].v10 = Math.Max(tree[l].v10 + tree[r].v10, tree[l].v11 + tree[r].v00);
        tree[x].v11 = Math.Max(tree[l].v10 + tree[r].v11, tree[l].v11 + tree[r].v01);
    }
}
