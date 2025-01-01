namespace ConsoleAppTest;

public class ProgramHard
{
    // 2972、统计移除递增子数组的数目2
    public long IncremovableSubarrayCount2(int[] nums)
    {
        int left = 0, nl = nums.Length;
        while (left < nl - 1)
        {
            if (nums[left] >= nums[left + 1])
                break;
            left++;
        }

        if (left == nl - 1)
            return 1L * nl * (nl + 1) / 2;
        
        long ans = left + 2;
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
}