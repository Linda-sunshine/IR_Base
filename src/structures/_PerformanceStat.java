package structures;

import utils.Utils;

public class _PerformanceStat {
	
	int[][] m_TPTable;
	
	double[][] m_prf; // Store the performance for each new prediction result.

	//Constructor for batch mode.
	public _PerformanceStat(int[][] TPTable){
		m_TPTable = TPTable;
		m_prf = new double[m_TPTable.length][3];//column: 0-1 class; row: precision, recall, F1.
	}
	
	public _PerformanceStat(int predL, int trueL){
		m_TPTable = new int[2][2];
		m_TPTable[predL][trueL]++;
		m_prf = new double[m_TPTable.length][3];//column: 0-1 class; row: precision, recall, F1.
	}
	
	public void addOnePredResult(int predL, int trueL){
		m_TPTable[predL][trueL]++;
	}
	
	public void calculatePRF(){
		for(int i=0; i<m_TPTable.length; i++){
			m_prf[i][0] = m_TPTable[i][i]/(Utils.sumOfRow(m_TPTable, i) + 0.00001);// Precision.
			m_prf[i][1] = m_TPTable[i][i]/(Utils.sumOfColumn(m_TPTable, i) + 0.00001); // Recall.
			m_prf[i][2] = 2*m_prf[i][0]*m_prf[i][1]/(m_prf[i][0] + m_prf[i][1] + 0.00001); // F1.
//			System.out.format("%d: %.4f\t%.4f\t%.4f\n", i, m_prf[i][0], m_prf[i][1], m_prf[i][2]);
		}
	}
	
	public void printTPPRF(){
		System.out.println("One user's TPTable.");
		for(int i=0; i<m_TPTable.length; i++){
			for(int j=0; j<m_TPTable[0].length; j++)
				System.out.print(m_TPTable[i][j] + "\t");
			System.out.println();
		}
		System.out.println("One user's prf.");
		for(int i=0; i<m_prf.length; i++){
			for(int j=0; j<m_prf[0].length; j++)
				System.out.print(m_prf[i][j] + "\t");
			System.out.println();
		}	
	}
	
	public double[][] getOneUserPRF(){
		return m_prf;
	}
	
	public int[][] getTPTable(){
		return m_TPTable;
	}
}
