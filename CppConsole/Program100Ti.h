//
// Created by z1772 on 24-9-20.
//

#ifndef PROGRAM100TI_H
#define PROGRAM100TI_H
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <bits/ranges_algo.h>
using namespace std;

class Program100Ti {
public:
    //1. 两数之和
    static vector<int> TwoSum(vector<int>& nums, int target) {
        unordered_map<int, int> map;
        for (auto i = 0; i < nums.size(); i++) {
            auto diff = map.find(target - nums[i]);
            if (diff != map.end()) {
                return {diff->second, i};
            }
            map.insert(pair(nums[i], i));
        }
        return {};
    }


    //49. 字母异位词分组
    static vector<vector<string>> GroupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;

        for (string& str: strs) {
            string key = str;
            ranges::sort(key);
            mp[key].emplace_back(str);
        }

        vector<vector<string>> ans;

        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.emplace_back(it->second);
        }

        return ans;
    }


    //128. 最长连续序列
    static int LongestConsecutive(vector<int>& nums) {
        if (nums.empty()) return 0;
        sort(nums.begin(), nums.end());
        int length = 1;
        int ans = 1;
        for (auto it = 1; it < nums.size(); ++it) {
            if (nums[it] == nums[it - 1] + 1) {
                length++;
                ans = max(ans, length);
            }else if (nums[it] > nums[it - 1]) {
                length = 1;
            }
        }
        return ans;
    }


    //283. 移动零
    static void moveZeroes(vector<int>& nums) {
        int nl = nums.size();
        int l = 0, r = 1;
        while (l < nl) {
            if (r == nl)
                break;

            if (nums[l]!=0)
                l++;

            if (nums[l] == 0 && nums[r] != 0)
            {
                nums[l] = nums[r];
                nums[r] = 0;
            }
            r++;
        }
    }


    //11. 盛最多水的容器
    static int MaxArea(vector<int>& height) {
        int l = 0, r = height.size() - 1;
        int ans = 0, nowArea = 0;
        while (l < r) {
            nowArea = min(height[l], height[r]) * (r - l);
            ans = max(ans, nowArea);
            if (height[l] > height[r]) {
                r--;
            }else {
                l++;
            }
        }
        return ans;
    }


    //15. 三数之和
    static vector<vector<int>> threeSum(vector<int>& nums) {
        
    }
};



#endif //PROGRAM100TI_H
