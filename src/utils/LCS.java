package utils;
/**
 * @author lingong
 * The algorithm for seeking the longest common subsequence(LCS) of two documents.
 */
public class LCS{
	int[][] m_c; //The array stores the number of longest subsequence of two string arrays.
	String[] m_x; //The first document to be computed.
	String[] m_y; //The second document to be computed.

	public void printLCS(int i, int j){
		if(i==0 || j==0)
			return;
		if(m_x[i-1] == m_y[j-1]){
			printLCS(i-1, j-1);
			System.out.print(m_x[i-1]);
		} else if(m_c[i-1][j]>=m_c[i][j-1])
			printLCS(i-1, j);
		else
			printLCS(i, j-1);
	}
	
	public int LCSLength(String[] x, String[] y){
		m_x = x;
		m_y = y;
		int m = m_x.length; 
		int n = m_y.length;
		m_c = new int[m+1][n+1];
		for(int i=1; i<=m; i++){
			for(int j=1; j<=n; j++){
				if(x[i-1] == y[j-1])
					m_c[i][j] = m_c[i-1][j-1]+1;
				else if(m_c[i-1][j] >= m_c[i][j-1]){
					m_c[i][j] = m_c[i-1][j];
				} else
					m_c[i][j] = m_c[i][j-1];
			}
		}
		return m_c[m][n];//Every cell contains the current longest common sequence of xi, yj.
	}
	public static void main(String[] args) {  
        String[] x = new String[]{"ab","bc","cd","B","D","A","B"};  
        String[] y = new String[] {"ac","cd","C","A","B","A"};  
        LCS test = new LCS();    
        System.out.println("The length is: " + test.LCSLength(x, y));  
        System.out.println("The LCS is:");  
        test.printLCS(x.length, y.length);
    }
}
