package mains;
import java.util.Random;

public class LCS{
	public void lcs(int i, int j, String[] x, int[][] b) {
		if (i == 0 || j == 0)
			return;
		if (b[i][j] == 1) {
			lcs(i - 1, j - 1, x, b);
			System.out.print(x[i] + "  ");
		} else if (b[i][j] == 2)
			lcs(i - 1, j, x, b);
		else
			lcs(i, j - 1, x, b);
	}

	private String[] init(String str) {
		String temp = str;
		String[] s = temp.split("");
		return s;
	}

	public int lcsLength(String[] x, String[] y, int[][] b) {
		int m = x.length - 1;
		int n = y.length - 1;

		int[][] c = new int[m + 1][n + 1];

		for (int i = 1; i <= m; i++) {
			c[i][0] = 0;
		}
		for (int i = 1; i <= n; i++)
			c[0][i] = 0;

		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				if (x[i].equals(y[j])) {
					c[i][j] = c[i - 1][j - 1] + 1;
					b[i][j] = 1;
				} else if (c[i - 1][j] >= c[i][j - 1]) {
					c[i][j] = c[i - 1][j];
					b[i][j] = 2;
				} else {
					c[i][j] = c[i][j - 1];
					b[i][j] = 3;
				}
			}
		}
		return c[m][n];
	}

	public static void main(String[] args) {  
        String s1 = "ABCBDAB";  
        String s2 = "BDCABA";  
        LCS cms = new LCS();  
        String[] x = cms.init(s1);  
        String[] y = cms.init(s2);  
        int[][] b = new int[x.length][y.length];  
  
        System.out.println("最大子序列的长度为："+cms.lcsLength(x, y, b));  
        System.out.println("最大子序列为：");  
        cms.lcs(x.length-1, y.length-1, x, b);
    } 
  
}
