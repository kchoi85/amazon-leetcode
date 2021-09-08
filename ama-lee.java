// =============
// ==== dfs ====
// =============

// # dfs 1 shell
/* (medium) !!!Asked by a lot
17. Letter combinations of a Phone Number
https://leetcode.com/problems/letter-combinations-of-a-phone-number/
*/
public List<String> letterCombinations(String digits) {
    List<String> result = new ArrayList<String>();
    if (digits == null || digits.length() == 0) return result;
    
    String[] mapping = {
    "0",
    "1",
    "abc",
    "def",
    "ghi",
    "jkl",
    "mno",
    "pqrs",
    "tuv",
    "wxyz"
};
    
    letterCombinationsRecursive(result, digits, "", 0, mapping);
    return result;
}
public void letterCombinationsRecursive(List<String> result, String digits, String current, int index, String[] mapping) {
    // base case to stop the recursive function
    if (index == digits.length()) {
        result.add(current);
        return;
    }

    String letter = mapping[digits.charAt(index) - '0'];
    for (int i=0; i<letter.length(); i++) {
        letterCombinationsRecursive(result, digits, current + letter.charAt(i), index + 1, mapping);
    }
}

// dfs 2 shell
// 733. Flood Fill
// https://leetcode.com/problems/flood-fill/

// dfs
class Solution {
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        if (image[sr][sc] == newColor) return image;

        // recursive f(x)
        fill(image, sr, sc, image[sr][sc], newColor);
        return image;
    }
    public void fill(int[][] image, int i, int j, int initialColor, int newColor) {
        // check if we are inside the bounds of the image?
        // Are pixels around the same color as the initialColor?
        if (i<0 || i >= image.length || j < 0 || j >=image[i].length || image[i][j] != initialColor) {
            return; 
        }
        image[i][j] = newColor;

        fill(image, i+1, j, initialColor, newColor); //Down
        fill(image, i-1, j, initialColor, newColor); //Up
        fill(image, i, j+1, initialColor, newColor); //Right
        fill(image, i, j-1, initialColor, newColor); //Left
    }
}

// # dfs 3 shell
// 695. Max Area of Island (med)
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int max = 0;
        for (int i=0; i<grid.length; i++) {
            for(int j=0; j<grid[i].length; j++) {
                // if 1 which we found island, we run dfs
                if (grid[i][j] == 1) {
                    max = Math.max(max, dfs(grid, i, j));
                }
            }
        }
        return max;
    }
    public int dfs(int[][] grid, int i, int j) {
        if (i<0 || i>=grid.length || j<0 || j>=grid[i].length || grid[i][j] == 0) {
            return 0;
        }

        grid[i][j] = 0;
        int count = 1;
        count += dfs(grid, i+1, j);
        count += dfs(grid, i-1, j);
        count += dfs(grid, i, j+1);
        count += dfs(grid, i, j-1);
        
        return count;
    }
}

// # dfs 4 shell (copy of List<Integer>) (tree Q)
// 113. Path Sum II (medium) // Combination sum II logic?
class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> paths = new ArrayList<>();
        findPaths(root, targetSum, new ArrayList<Integer>(), paths);
        return paths;
    }
    public void findPaths(TreeNode root, int targetSum, List<Integer> current, List<List<Integer>> paths) {
        if (root == null) return;
        
        current.add(root.val);
        
        if (root.left == null && root.right == null && targetSum - root.val == 0) {
            paths.add(current);
            return;
        }
        findPaths(root.left, targetSum - root.val, new ArrayList<Integer>(current) , paths);
        findPaths(root.right, targetSum - root.val, new ArrayList<Integer>(current) , paths);
        
    }
} //O(N)


// # dfs 5 shell
// 286. Walls and Gates
// start at gate (0) and pass count into recursive dfs which means wherever we go we incrase 
// count by 1 and assign the that position
class Solution {
    public void wallsAndGates(int[][] rooms) {
        for (int i=0; i<rooms.length; i++) {
            for (int j=0, j<rooms[i].length; j++) {
                if (rooms[i][j] == 0) {
                    dfs(i,j,0,rooms);
                }
            }
        }
    }
    public void dfs(int i, int j, count, int[][] rooms) {
        if (i<0 || i>=rooms.length || j<0 || j>=rooms[i].length, rooms[i][j] < count) {
            return;
        }
        rooms[i][j] = count;
        dfs(i+1, j, count+1, rooms);
        dfs(i-1, j, count+1, rooms);
        dfs(i, j+1, count+1, rooms);
        dfs(i, j-1, count+1, rooms);
    }
}

// #dfs 7 shell
// 79. Word search (med)
class Solution {
    public boolean wordSearch(char[][] board, String word) {
        for (int i=0; i<board.length; i++) {
            for (int j=0; j<board[i].length, j++) {
                if (board[i][j] == word.charAt(0) && dfs(board, i, j, 0, word)) {
                    return true;
                } 
            }
        }
        return false;
    }
    public boolean dfs(char[][] board, int i, int j, int index, String word) {
        if (index == word.length()) return true;
        if (i<0||i>=board.length||j<0||j>=board[i].length|| board[i][j] != word.charAt(index)) {
            return false;
        }
        // haven't found the full word, haven't gone out of board and is charAt(count)
        // since can't use previously used cell
        char temp = board[i][j];
        board[i][j] = ' ';

        boolean found = dfs(board, i + 1, j, index + 1, word)
            || dfs(board, i-1, j, index + 1, word)
            || dfs(board, i, j+1, index + 1, word)
            || dfs(board, i, j-1, index + 1, word)
        
        board[i][j] = temp;
        
        return found;
    }
} // O(M*N), 
  // space: modifying the board, and recursive, and worse case (if we need to look at all char in board) -> O(N)

// # dfs 8 shell (add or remove simulation logic)
// 78. Subsets
// recursive call (take or not take)
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        generateSubsets(nums, 0, new ArrayList<Integer>(), result);
        return result;
    }
    public void generateSubsets(int[] nums, int index, List<Integer> current, List<List<Integer>> result) {
        result.add(new ArrayList<>(current));
        for (int i=index; i<nums.length; i++) {
            current.add(nums[i]);
            generateSubsets(nums, i+1, current, result);
            current.remove(current.size() - 1);
        }
    }
} // runtime: O(2N) N=number of elements in nums, and 2 since we have choice to take or not take
// space: O(N) -> as deep as the recursive function can go

// WTF?
// # dfs 6 shell (add and remove simulation logic) // same as dfs # 8 Subsets
// 40. Combination sum II (med)
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        findCombination(candidates, target, 0, new ArrayList<Integer>(), result);
        return result;
    }
    public void findCombination(int[] candidates, int target, int index, List<Integer> current, List<List<Integer>> result) {
        if (target == 0) {
            result.add(new ArrayList<>(current));
            return;
        }
        if (target < 0) {
            return;
        }
        
        for (int i=index; i<candidates.length; i++) {
            if (i == index || candidates[i] != candidates[i-1]) {
                current.add(candidates[i]);
                findCombination(candidates, target - candidates[i], i + 1, current, result);
                current.remove(current.size() - 1);
            }
        }
    }
} // runtime O(2N) 
  //space O(N) -> all of our recursive calls are going as deep as candidates number 

// =============================================================
// ===============================end of dfs   
// =============================================================



// ===============
// ==== TREES ==== BST
// ===============
/*
112. Path Sum
GIven binary tree and a sum, determine if the tree has a root-to-leaf path 
sums to the sum (22) -> true (5->4->11->2)
       5  
    4    8
  11  13  4
7   2       1
*/ 

// case 1, arrive @ leaf node child == null, false
// case 2, arrive @ leaf node and sum - root.val == 0, true
// otherwise (else) recursive call to left or right to see if true
public boolean hasPathSum(TreeNode root, int sum) {
    // case where we've reached the leaf node's child (which means no path)
    if (root == null) {
        return false;
    } else if (root.right == null && root.left == null && sum - root.val == 0) {
        return true; // case where we are at leaf node and sum - root.val = 0
    } else { // recursive call left and right, subtract the sum - root.val
             // either can return true
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val)
    }
}

// 100. Same Tree
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p==null && q==null) return true;
        else if (p==null || q==null) return false;
        else if (p.val != q.val) return false;
        else {
            return isSameTree(p.left, q.left) && isSameTreef(p.right, q.right);
        }
    }
} // traverse entire tree of p and q, -> same amount of nodes -> O(N)

// 101. Symmetric Tree (balanced)
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return false;
        return isSymRecursive(root.left, root.right);
    }
    public boolean isSymRecursive(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        else if (left == null || right == null) return false;
        else if (left.val != right.val) return false;
        else {
            return isSymRecursive(left.left, right.right) && isSymRecursive(left.right, right.left);
        }
    }
} // O(N)

// 572. Subtree of another tree
class Solution {
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if (s==null) return false; //as if t can't be a null s tree
        else if (isSameTree(s, t)) return true;
        else {
            return isSubtree(s.left, t) || isSubtree(s.right , t);
        }
    }
    public boolean isSameTree(TreeNode s, TreeNode t) {
        if (s == null && t == null) return true;
        else if (s == null || t == null) return false;
        else if (s.val != t.val) return false;
        else {
            return isSameTree(s.left, t.left) && isSameTree(s.right, t.right);
        }
    }
} // O(MN), space: minimum of M bw N, O(minimum between M and N)

/* 
235. Lowest Common Ancestor of a BST
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
*/
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    // check if p and q are either on the node's left or node's right, if not
    // it's a split (children)
    if (p.val < root.val && q.val < root.val) { // both on left
        return lowestCommonAncestor(root.left, p, q);
    } else if (p.val > root.val && q.val > root.val) {
        return lowestCommonAncestor(root.right, p, q);
    } else {
        return root;
    }
 } //Time complexity is O(h), where h is the height of the tree
 // space O(h)

 // 236. Lowest Common Ancestor of a Binary Tree
 class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) return right;
        else if (right == null) return left;
        else return root;

    }
}

// 104. Maximum depth of BST
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int left_depth = maxDepth(root.left) + 1;
        int right_depth = maxDepth(root.right) + 1;
        return Math.max(left_depth, right_depth);
    }
}

// 199. Binary Tree Right Side View (says top to bottom -> use BFS)
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        
        while (!queue.isEmpty()) {
            int size = queue.size();
            
            while (size-- > 0) {
                TreeNode current = queue.remove();
                if (size == 0) {
                    result.add(current.val);
                }
                if (current.left != null) {
                    queue.add(current.left);
                }
                if (current.right != null) {
                    queue.add(current.right);
                }
            }
        }
        return result;
    }
} // time: go through entire tree, O(N), where N is the number of nodes in tree
// space = O(N)

// 103. binary tree zigzag
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            for (int i=0; i<size; i++) {
                TreeNode current = queue.remove();
                currentLevel.add(current.val);
                if (current.left != null) {
                    queue.add(current.left);
                }
                if (current.right != null) {
                    queue.add(current.right);
                }                
            }
            result.add(currentLevel);
        }
        return result;
    }
}

// 938. Range Sum of BST (BFS)
class Solution {
    public int rangeSum(TreeNode root, int L, int R) {
        int sum = 0;
        if (root == null) return sum;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);

        while(!queue.isEmpty()) {
            TreeNode current = queue.remove();
            if (current.val >= L && current.val <= R) {
                sum += current.val;
            }
            // if root has left child and that val is > L
            if (current.left != null && current.val > L) {
                queue.add(current.left);
            }
            if (current.right != null && current.val < R) {
                queue.add(current.right);
            }
        }
        return sum;
    }
}

// 257. binary tree paths
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> paths = new ArrayList<>();
        if (root == null) return paths;
        
        dfs(root, "", paths);
        return paths;
    }
    public void dfs(TreeNode root, String current, List<String> paths) {
        // beginning of the treenode
        current += root.val;
        
        // end of the treenode (leaf)
        if (root.left == null && root.right == null) {
            paths.add(current);
            return;
        }
        
        if (root.left != null) {
            dfs(root.left, current + "->", paths);
        }
        if (root.right != null) {
            dfs(root.right, current + "->", paths);
        }
    }
} // O(N) N=height of tree, space O(N), recursive stack can go deep as tree

// validate binary tree
class Solution {
    public boolean isValidsBST(TreeNode root) {
        // take root, move down root, (left subtree is less than root.val, and right subtree > root)
        // traverse the tree, and all the way past leaf = valid BST
        // constraints not met = false

        return validate(root, null, null); // max, min, initially we don't know if we have root
    }
    public boolean validate(TreeNode root, Integer max, Integer min) { // Integer object!
        // base case
        if (root == null) return true; // we've went all the way past leaves
        else if (max != null && root.val >= max || min != null && root.val <= min) return false; 
        // above: checking if root.val violates the constraints of BST
        else {
            // recursive calls
            return validate(root.left, root.val, min) && validate(root.right, max, root.val);
            // we pass root.left, and the max as root.val since left subtree is always smaller, therefore root.val should be max
            // similarly, on the right subtree, root.val should be the minimum
        }
    }
} // O(N), space: O(N), worse case massive linkedlist 

// sum of left leaves
class Solution {
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) return 0;
        else if (root.left != null && root.left.left == null && root.left.right == null) {
            return root.left.val + sumOfLeftLeaves(root.right);
        } else {
            return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
        }
    }
}

// ===========================================================================
// ===========================================================================end of bst
// ===========================================================================



// ============================================================
// ==== dp ======== dp ======== dp ======== dp ======== dp ====
// ========================================================================

/* 
70. Climbing Stairs
In: 2
Out: 2
(1+1) (2)
*/
public int climbingStairs(int n) {
    int[] dp = new int[n + 1]; // dp of 0 = 0; number of ways to climb 0 stairs is 1 way (none)
    dp[0] = 1;
    dp[1] = 1;
    for (int i=2; i<=n; i++) { // up to and including the nth step
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
}

// 62. Unique Paths (Medium) dp
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i=0; i<dp.length; i++) {
            dp[i][0] = 1; // for every row we assign 1 by going down
        }
        for (int i=0; i<dp[0].length; i++) {
            dp[0][i] = 1; // for every column in the 0th row, we assign 1
        }

        // this means that for any cell, we must have come from above or from the left
        // since 0th row and 0th column is already 1,
        for (int i=1; i<dp.length; i++) {
            for (int j=1; j<dp[i].length; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1]; // means we add from top and left
            }
        }
        return dp[m-1][n-1];
    }
}

// 198. House Robber (bottom up processing - dp)
class Solution {
    public int robHouse(int[] nums) {
        // case: 0 house
        if (nums == null || nums.length == 0) return 0;
        // case: 1 house, return nums[0]
        if (nums.length == 1) return nums[0];
        // case 2: 2 houses, rob the house that has more money
        if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        }
        // case 3+: 3 houses, do we rob 1st and 3rd or 2nd? (bottom up)
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i=2; i<dp.length; i++) {
            dp[i] = Math.max(nums[i] + dp[i-2], dp[i-1]);
        }
        return dp[nums.length - 1];
    }
}

// eh wtf
// 322. Coin change (med)
// coins = [1,2,5], amount = 11
// output = 3, (5+5+1=11) / -> if not exist return -1;
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount +1);

        dp[0] = 0;

        for (int i=0; i<=amount; i++) {
            for (int j=0; j<coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }        
        return dp[amount] > amount ? -1 : dp[amount]; 
    }
} // time: (NM), space: (N)

// 64. minimum path sum (med)
class Solution {
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        int[][] dp = new int[grid.length][grid[0].length];
        for (int i=0; i<grid.length; i++) {
            for (int j=0; j<grid[i].length; j++) {

                dp[i][j] = grid[i][j];
                
                if (i>0 && j>0) {
                    dp[i][j] += Math.min(dp[i-1][j], dp[i][j-1]);
                } else if (i>0) {
                    dp[i][j] += dp[i-1][j];
                } else if (j>0) {
                    dp[i][j] += dp[i][j-1];
                }
            }
        }
        return dp[grid.length - 1][grid[0].length - 1];
    }
}


// O(MN) && O(MN);


//=================================
//============ end of dp ===============
//=================================

/* 
387. First Unique Character in a String
s = "leetcode" -> return 0
s = "loveleetcode" -> 2
*/
public int firstUniqueChar(String s) {
    HashMap<Character, Integer> map = new HashMap<Character, Integer>();
    for (int i=0; i<s.length(); i++) {
        char current = s.charAt(i);
        if (!map.containsKey(current)) {
            map.put(current, i);
        } else {
            map.put(current, -1);
        }
    }

    int min = Integer.MAX_VALUE;
    for (char c: map.keySet()) {
        if (map.get(c) > -1 && map.get(c) < min) {
            min = map.get(c);
        }
    }
    return min == Integer.MAX_VALUE ? -1 : min;
} // Time complexity: O(N) since we go through the string length of N

/*
136. Single Number
Given a non-empty array of integers, every element appears twice except for one.
Input: [2,2,1]
Output: 1
*/
public int singleNumber(int[] nums) {
    HashSet<Integer> set = new HashSet<Integer>():
    for (int num: nums) {
        if (!set.contains(num)) {
            set.add(num);
        } else {
            set.remove(num);
        }
    }

    for (int i: set) {
        return i;
    }
    return -1;
} // TC: O(N) since we go thoruhg the length of nums array

/*
231. Power of Two
Given an integer, write a function to determine if it is a power of two
Input: 1
Output: true
Input: 16
Output: true (2^4)
*/
public boolean isPowerOfTwo(int n) {
    long i = 1; // so we don't overflow
    while (i < n) {
        i *= 2;
    }
    return i == n;
} // O(N)

/*
657. Robot Return to Origin
"UD" -> true
"LL" -> false
*/
public boolean judgeCircle(String moves) {
    int UD = 0;
    int LR = 0;
    for (int i=0; i<moves.length(); i++) {
        char current = moves.charAt(i);
        if (current == 'U') {
            UD++;
        } else if (current == 'D') {
            UD--;
        } else if (current == 'L') {
            LR++;
        } else if (current =='R') {
            LR--;
        }
    }
    if (UD == 0 && LR == 0) {
        return true;
    }
    return false;
} // O(N)

/*
268. Missing Number
[3,0,1] -> 2
*/
// When trying to remember we've seen, HashSet is good
public int missingNumber(int[] nums) {
    HashSet<Integer> set = new HashSet<Integer>();
    for (int i: nums) {
        set.add(i);
    }    
    for (int i=0; i<= nums.length; i++) {
        if (!set.contains(i)) {
            return i;
        }
    }
    return -1;
} // O(N)

/*
122. Best Time to Buy and Sell Stock II
[7,1,5,3,6,4] -> 7
Buy on day 2 (price=1) and sell on day 3 (price=5), profit = 5 - 1 = 4
Then buy on day 4 (price=3) and sell on day 5 (price=6), profit = 6 - 3 = 3
*/
public int maxProfit(int[] prices) {
    if (prices == null || prices.length == 0) return 0;

    int profit = 0;
    for (int i=0; i<prices.length - 1; i++) {
        if (prices[i+1] > prices[i]) {
            profit += prices[i+1] - prices[i];
        }
    }
    return profit;
}

/*
27. Remove Element
Given array and value, remove all instances of the value from array
and return the length of the array
[0,1,2,2,3,0,4,2], val=2, -> length = 5
*/
public int removeElement(int[] nums, int val) {
    int index = 0;
    for (int i: nums) {
        if (i != val) {
            nums[index++] = i;
        }
    }
    return index; // which keeps the length of the new array
}

// 771. Jewels and Stones
// Given J and S, find how many jewels you have
// J=aA, s=aAAbbbb -> 3
public int jewelsStones(String J, String S) {
    HashSet<Character> jewels = new HashSet<Character>();
    for (char c: J.toCharArray()) {
        jewels.add(c);
    }

    int numJewels = 0;
    for (char c: S.toCharArray()) {
        if (jewels.contains(c)) {
            numJewels++;
        }
    }
    return numJewels;
} // O(M+N)

// 11. Container with Most Water
class Solution {
    public int maxArea(int[] height) {
        int a_pointer = 0;
        int b_pointer = height.length - 1;
        
        int maxA = 0;
        while (a_pointer < b_pointer) {
            if (height[a_pointer] < height[b_pointer]) {
                maxA = Math.max(maxA, height[a_pointer] * (b_pointer-a_pointer));
                a_pointer++;
            } else {
                maxA = Math.max(maxA, height[b_pointer] * (b_pointer-a_pointer));
                b_pointer--;
            }
        }
        return maxA;
    }
}



// 206. Reverse Linked List
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        
        while (current != null) {
            ListNode temp = current.next;
            current.next = prev;
            prev = current;
            current = temp;   
        }
        
        return prev;
    }
}

// 819. Most common word
class Solution {
    public String mostCommonWord(String paragraph, String[] banned) {
        HashSet<String> bannedWords = new HashSet<>();
        for (String word: banned) {
            bannedWords.add(word);
        }

        HashMap<String, Integer> counts = new HashMap<>();
        for (String word: paragraph.replaceAll("[^a-zA-Z]", " ").toLowerCase().split(" ")) {
            if (!bannedWords.contains(word)) {
                counts.put(word, counts.getOrDefault(word, 0) + 1);
            }
        }

        String result = "";
        for (String word: counts.keySet()) {
            if (result.equals("") || counts.get(word) > counts.get(result)) {
                result = word;
            }
        }
        return result;
    }
}

// 929. Unique Email Address
// https://leetcode.com/problems/unique-email-addresses/
class Solution {
    public int numUniqueEmail(String[] emails) {
        HashSet<String> set = new HashSet<String>();
        for (String email: emails) {
            StringBuilder address = new StringBuilder();
            for (int i=0; i<email.length(); i++) {
                char c = email.charAt(i);
                if (c == '.') {
                    continue;
                } else if (c =='+') {
                    while (email.charAt(i) != '@') {
                        i++;
                    }

                    address.append(email.substring(i + 1));
                } else {
                    address.append(c);
                }
            }

            set.add(address.toString())
        }
        return set.size();
    }
}

// 253. Meeting Rooms II
// https://www.youtube.com/watch?v=PWgFnSygweI&list=PLi9RQVmJD2fZgRyOunLyt94uVbJL43pZ_&index=18
// 0 30 , 5 10, 15 20
// 2 5, 7 10
// 2 5, 3 10
class Solution {
    public int minMeetingRooms(Interval[] intervals) {
        if (intervals == null || intervals.length == 0) return 0;

        // lambda function to sort meetings by start time
        Arrays.sort(intervals, (a, b) -> a.start - b.start);

        // minHeap (compare objects, keep tracking of end times)
        // root of the minHeap = meeting that ends the earliest
        PriorityQueue<Interval> minHeap = new PriorityQueue<>((a,b) -> a.end - b.end); // root contains teh meeting that ends the earliest
        minHeap.add(intervals[0]);

        for (int i=1; i<intervals.length; i++) {
            Interval current = intervals[i];
            Interval earliest = minHeap.remove();
            if (current.start >= earliest.end) { // >= edge case to consider ending at 2 and starting a new meeting at 2
                earliest.end = current.end;
            } else {
                minHeap.add(current);
            }
            minHeap.add(earliest);
        }
        return minHeap.size();
    }
} //O(N*Log(N))

// Most asked AMAZON QUESTION !!!
// 973. K Closest Points to Origin

/*
Need data structure: MaxHeap
use maxheap to make sure the size of max heap is of size K
when we add to max heap but the size is already K, then we remove the next thing in the heap
(removes the maximum, which is the furthest away from origin)
*/
class Solution {
    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>((a,b) -> b[0] * b[0] + b[1] * b[1] - (a[0] * a[0] + a[1] * a[1]));
        for (int[] point: points) {
            maxHeap.add(point);
            while (maxHeap.size() > k) {
                maxHeap.remove();
            }
        }
        int[][] result = new int[k][2];
        while (k-- > 0) {
            result[k] = maxHeap.remove();
        }
        return result;
    }
}
// int[][] result = new int[k][2];
// while (k-- > 0) {
//     result[k] = maxHeap.remove();
// }

class Solution {
    public List<List<Integer>> kClosest(List<List<Integer>> points, int k) {
        PriorityQueue<List<Integer>> maxHeap = new PriorityQueue<>((a, b) -> b[0] * b[0] + b[1] * b[1] - (a[0] * a[0] + a[1] * a[1]));
        for (List<Integer> point: points) {
            maxHeap.add(point);
            while (maxHeap.size() > k) {
                maxHeap.remove();
            }
        }
        List<List<Integer>> result = new ArrayList<>();
        while (maxHeap.size() > 0) {
            result.add(maxHeap.remove());
        }
        return result;
    }
}



// 403. Frog Jump (Hard)
class Solution {
    public boolean canCross(int[] stones) {
        
    }
}

// 23. Merge K Sorted Lists (hard)
//  throw every list into minHeap, make a new list from the heap
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        // for every list, throw every node into minheap
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (ListNode head: lists) {
            while (head != null) {
                minHeap.add(head.val);
                head = head.next;
            }
        }
        ListNode dummy = new ListNode(-1); // dummy to place hold -1
        // head point to the dummy (reference to the head to add to that list)
        ListNode head = dummy;
        while (!minHeap.isEmpty()) { // minHeap.size() > 0
            head.next = new ListNode(minHeap.remove()); // remember that minheap stored integer val
            head = head.next;
        }

        return dummy.next;
    }
}

// 49. Grouped Anagrams (medium) !!! frequent
// Input: strs = ["eat","tea","tan","ate","nat","bat"]
// Output: 
//     [
//         ["bat"],
//         ["nat","tan"],
//         ["ate","eat","tea"]
//     ]
class Solution {
    public List<List<String>> groupedAnagrams(String[] strs) {
        List<List<String>> groupedAnagrams = new  ArrayList<>();
        HashMap<String, List<String>> map = new HashMap<>();

        for(String current: strs) {
            char[] characters = current.toCharArray();
            Arrays.sort(characters);
            String sorted = new String(characters);
            if (!map.containsKey(sorted)) {
                map.put(sorted, new ArrayList<>());
            }

            map.get(sorted).add(current);
        }

        for (List<String> val: map.values()) {
             groupedAnagrams.add(val);
        }
        return groupedAnagrams;
    }
}

// 20210906

// #31 
// 905. Sort array by parity (even in front, and odd at back)
class Solution {
    public int[] sortArrayByParity(int[] nums) {
        int a_pointer = 0;
        int b_pointer = nums.length - 1;
        while (a_pointer < b_pointer) {
            // case where nums[a_pointer] is odd
            if (nums[a_pointer] % 2 > nums[b_pointer] % 2) {
                int temp = nums[a_pointer];
                nums[a_pointer] = nums[b_pointer];
                nums[b_pointer] = temp;
            }
            
            if (nums[a_pointer] % 2 == 0) a_pointer++;
            if (nums[b_pointer] % 2 == 1) b_pointer--;
        }
        return nums;
    }
} //O(N), space comp = O(1) constant

// 345. Reverse vowels of a string
class Solution {
    public String reverseVowels(String s) {
        // O(1) to determine if something is vowel
        HashSet<Character> set = new HashSet<>();
        set.add('a');
        set.add('e');
        set.add('i');
        set.add('o');
        set.add('u');
        set.add('A');
        set.add('E');
        set.add('I');
        set.add('O');
        set.add('U');
        
        char[] character = s.toCharArray();

        int a_pointer = 0;
        int b_pointer = s.length() - 1;
        
        while (a_pointer < b_pointer) {
            if (set.contains(character[a_pointer]) && set.contains(character[b_pointer])) {
                char temp = character[a_pointer];
                character[a_pointer] = character[b_pointer];
                character[b_pointer] = temp;
                
                // sort by parity have cases at the 2 if where if they are even/odd, the 2 pointer are in/decremented
                // here, we only check if it's not a vowel so we need to manually in/decrement it
                a_pointer++;
                b_pointer--;
            }
            
            if (!set.contains(character[a_pointer])) {
                a_pointer++;
            }
            if (!set.contains(character[b_pointer])) {
                b_pointer--;
            }
        }
        return new String(character);
    }
}





// WTF?
//  763. Partition Labels (med)
class Solution {
    public List<Integer> partitionLables(String s) {
        List<Integer> paritionLenghts = new ArrayList<>();
        int[] lastIndex = new int[26]; //a-z
        for (int i=0; i<s.length(); i++) {
            lastIndex[s.charAt(i) - 'a'] = i; // -'a' gets the index
        }

        int i=0;
        while (i<s.length()) {
            int end = lastIndex[s.charAt(i) - 'a'];
            int j = i;
            while (j != end) {
                end = Math.max(end, lastIndex[s.charAt(j++) - 'a']);
            }
            paritionLenghts.add(j - i + 1);
            i = j + 1;
        }
        return partitionLables;
    }
} // ruintime O(N), n is number of characters in string, space: o(1), n = no of char

//  14. Longest Common Prefix
class Solution {
    public String longestCommonPrefix(String[] strs) {
        String longestCommonPrefix = "";
        if (strs == null || strs.length == 0) return longestCommonPrefix;

        // check if all strings have the same 0
        int index = 0;
        // iterate over the first string
        for (char c: strs[0].toCharArray()) {
            // compare with second and onward
            for (int i=1; i<strs.length; i++) {
                // case where index >= next string length or c != nextstring char at
                if (index >= strs[i].length() || c != strs[i].charAt(index)) {
                    return longestCommonPrefix;
                }
            }
            longestCommonPrefix += c;
            index++;
        }
        return longestCommonPrefix;
    }
} // going over all characters in first string, comparing to all char is all other string
// going over every single character in every given string -> O(N)
// space = O(1) constant space



// 21. Merge 2 sorted list
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        while (l1 != null) {
            minHeap.add(l1.val);
            l1 = l1.next;
        }
        while (l2 != null) {
            minHeap.add(l2.val);
            l2 = l2.next;
        }
        
        ListNode dummy = new ListNode(-1); 
        ListNode head = dummy;
        while (minHeap.size() > 0) {
            head.next = new ListNode(minHeap.remove());
            head = head.next;
        }
        return dummy.next;
    }
} // O(Nlog*k) where N is the total number of elements among all the linkedlists, k=number of lists
// Auxiliary Space: O(k). 
// The priority queue will have atmost ‘k’ number of elements at any point of time, hence the additional space required for our algorithm is O(k).





// 1119. Remove vowels from a string
class Solution {
    public String removeVowels(String S) {
        HashSet<Character> set = new HashSet<>();
        set.add('a');
        set.add('e');
        set.add('i');
        set.add('o');
        set.add('u');

        StringBuilder result = new StringBuilder(); // build fast and more efficient

        for (char c: S.toCharArray()) {
            if (!set.contains(c)) {
                result.append(c);
            }
        }
        return result.toString();
    }
}




// 160. intersection of two linked lists
class Solution {
    public ListNode = getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        
        // pointers to traverse the ListNodes
        ListNode a_pointer = headA;
        ListNode b_poiunter = headB;

        while (a_pointer != b_pointer) {
            if (a_pointer == null) { // when a_pointer reachers to the end of the list, we set to headB
                a_pointer = headB;
            } else {
                a_pointer = a_pointer.next; // if not end of the list, continue
            }

            if (b_pointer == null) { // when a_pointer reachers to the end of the list, we set to headB
                b_pointer = headA;
            } else {
                b_pointer = b_pointer.next; // if not end of the list, continue
            }
        }
        return a_pointer;
    }
}

//20210907

// 1046. Last Ston weight
class Solution {
    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>();
        for (int stone: stones) {
            maxHeap.add(-stone);
        }

        while (maxHeap.size() > 1) { // !!! important when remove two from heap!
            int stoneOne = -maxHeap.remove();
            int stoneTwo = -maxHeap.remove();
            if (stoneOne != stoneTwo) {
                maxHeap.add(-(stoneOne - stoneTwo));
            } 
        }
        return maxHeap.size() == 0 ? 0 : -maxHeap.remove();
    }
}

// minimum cost to connect sticks
public class Solution {
    /**
     * @param sticks: the length of sticks
     * @return: Minimum Cost to Connect Sticks
     */
    public int MinimumCost(List<Integer> sticks) {
        // write your code here
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int i=0; i<sticks.size(); i++) {
            minHeap.add(sticks.get(i));
        }

        int count = 0;
        while (minHeap.size() > 1) { // !!! important when remove two from heap!
            int sum = 0;
            sum = minHeap.remove() + minHeap.remove();
            count += sum;
            minHeap.add(sum);
        }
        return count;
    }
}

// 415. add strings
class Solution {
    public String addStrings(String num1, String num2) {
        int a_pointer = num1.length() - 1;
        int b_pointer = num2.length() - 1;
        int carry = 0;
        
        StringBuilder result = new StringBuilder();
        
        while (a_pointer >= 0 || b_pointer >= 0) {
            int sum = carry;
            if (a_pointer >= 0) {
                sum += num1.charAt(a_pointer--) - '0';
            }
            if (b_pointer >= 0) {
                sum += num2.charAt(b_pointer--) - '0';
            }
            
            result.append(sum % 10);
            carry = sum / 10;
            
        }
        if  (carry != 0) {
            result.append(carry);
        }
        return result.reverse().toString();
    }
}

// 1007. minimum domino rotations for equal row (med)
class Solution {
    public int minDom(int[] A, int[] B) {
        int minSwaps = Math.min(
            numSwaps(A[0], A, B), // make everything in A match A[0] (how many times for everything in A to match A[0])
            numSwaps(B[0], A, B) // make everything in A match B[0]
        )

        minSwaps = Math.min(minSwaps, numSwaps(A[0], B, A)); // make everything in B match A[0]
        minSwaps = Math.min(minSwaps, numSwaps(B[0], B, A)); // make everything in B match B[0]
    
        return minSwaps == Integer.MAX_VALUE ? -1 : minSwaps;
    }
    public int minSwaps(int target, int[] A, int[] B) {
        int numSwaps = 0;
        for (int i=0; i<A.length, i++) {
            if (A[i] != target && B[i] != target) { // where swapping wil never equal the target
                return Integer.MAX_VALUE;
            } else if (A[i] != target) {
                numSwaps++;
            }
        }
        return numSwaps;
    }
}

// 767. Reorganize string (med)
class Solution {
    public String reorg(String S) {
        HashMap<Character, Integer> map = new HashMap<>();
        for (char c: S.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        // order by how many times it occured
        PriorityQueue<Character> maxHeap = new PriorityQueue<>((a,b) -> map.get(b) - map.get(a));
        
        // add all keys
        maxHeap.add(map.keySet());

        StringBuilder result = new StringBuilder(); // string manipulation is expensive in java
        while (maxHeap.size() > 1) {
            char current = maxHeap.remove();
            char next = maxHeap.remove();
            result.append(current);
            result.append(next);
            
            map.put(current, map.get(current) - 1);
            map.put(next, map.get(next) - 1);

            if (map.get(current) > 0) {
                maxHeap.add(current);
            }
            if (map.get(next) > 0) {
                maxHeap.add(next);
            }
        }

        if (!maxHeap.isEmpty()) {
            char last = maxHeap.remove();
            if (map.get(last) > 1) {
                return "";
            }
            // if only 1 left, than we add to the result
            result.append(last);
        }
        return result.toString();
    }
}

// 735. asteroid colision
class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        int i = 0;
        while (i<asteroids.length) {
            // if positive integer (going to right), throw into stack
            if (asteroids[i] > 0) {
                stack.push(asteroids[i]);
            } else { // negative
                // whatever we collid is whatever is on top of stack
                while (!stack.isEmpty() && stack.peek() > 0 && stack.peek() < Math.abs(asteroids[i])) {
                    stack.pop();
                }
                if (stack.isEmpty() || stack.peek() < 0) {
                    stack.push(asteroids[i]);
                } else if (stack.peek() == Math.abs(asteroids[i])) {
                    stack.pop();
                }
            }
            i++;
        }

        int[] remaining = new int[stack.size()];
        for (i = stack.size() - 1; i>=0; i--) {
            remaining[i] = stack.pop();
        }

        return remaining;
    }
} // walking throuhg asteroids, O(N), N=# of asteroids, space=O(N), stack holding all items/most

// 443. string compression
// solve using O(1), so can't store anything in memory (no DS)
class Solution {
    public int compress(char[] chars) {
        int index=0;
        int i=0;
        while (i<chars.length) {
            int j = i;
            chars[index++] = chars[i];

            while (j<chars.length && chars[j] == chars[i]) {
                j++;
            }
            if (j - i > 1) {
                String count = j - i + "";
                for (char c: count.toCharArray()) {
                    chars[index++] = c;
                }
            }
            i = j;
        }
        return index;
    }
} // O(N) since we use pointer, O(1)

// remove duplicates from SORTED array
class Solution {
    public int removeDup(int[] nums) {
        int index = 1;
        for (int i=0; i<nums.length - 1; i++) {
            if (nums[i] != nums[i+1]) {
                nums[index++] = nums[i + 1];
            }
        }
        return index;
    }
}

// https://leetcode.com/problems/maximum-units-on-a-truck/
class Solution {
    public int maximumUnits(int[][] boxTypes, int truckSize) {
        Arrays.sort(boxTypes, (a,b) -> b[1] - a[1]); 
        int maxSize = 0;
        for (int[] box: boxTypes) {
            if (truckSize < box[0]) {
                return maxSize + truckSize * box[1];
            }
            maxSize += box[0] * box[1];
            truckSize -= box[0];
        }
        return maxSize;
    }
}

//jump game
class Solution {
    public boolean canJump(int[] nums) {
        int lastIndex = nums.length - 1;
        for (int i = nums.length - 1; i>=0; i--) {
            if (i + nums[i] >= lastIndex) {
                lastIndex = i;
            }
        }
        return lastIndex == 0;
    }
} //greedy, O(N), O(1)

/*https://www.geeksforgeeks.org/amazon-interview-experience-for-sde-1-14/

 A DFS graph question, where we want to identify paths of length K from the start to end of a DAG. Probably LC Easy
 //  maximum duplicate string
//  One question asked from amongst various topics like Graph,
// !!!Graphs and Trees
//  bfs for a graph
// Tree DFS to find average employee rating based on chain of command.

//best time for using a washing machine for minimum electricity
// Amazon Truck Fill coding question
// 2. Convert integer to Roman.
// 3. Create a function (wrapped in a class) that mimics the find command in Linux
// 4. Given a list of words, determine which words can be decomposed.

//OOD
 //Design a system that looks for a certain file.
 //Design an executor for tasks with dependencies.
 Q. Design an algorithm to efficiently place amazon packages in the lockers
Q. Design a library catalog system
Q. Gather logs and append logs to a file sorted by start time
Q. Find major version conflicts during package building
lc 128 + OOP(design a util class to search file recursively, make sure the extendability, using abstract base class and depth-first search)

// Discuss, how to store a whole database (comprising of at least 1 primary key) in a data structure. Needed to focus on the lookup time via the key provided.

Q: Tell me about a time when you went above and beyond.
Q: Tell me about a time you had a conflict
Q: Tell me about a time when you felt you did not have enough Subject Matter Expertise and how did you go about it
Tell me about a time when you failed on a significant project
Tell me about a time you made a bad decision that had a significant impact.
Think of a time when you delivered successfully on a tight deadline.
Q: Did you make any mistakes and what did you learn from it?
Comment about a time where you took more responsibility.
Comment about a time where you failed a commitment.
Tell me about a time when you had to make a quick decision that was going to have a significant impact on the organization?



*/




