package structures;

import utils.Utils;

public class _PerformanceStat {
	int m_totalNo;
	double[][] m_rawStat;//This stat stores the raw data of predicted reviews.
	double[][][] m_prfStat; // Store the performance for each new prediction result.
	
	public _PerformanceStat(double[][] raw){
		m_rawStat = raw;
		m_totalNo = raw.length;
	}
	
	public void calcuatePreRecF1(){
		int[][] TPTable = new int[2][2];
		m_prfStat = new double[m_totalNo][][];
		for(int i=0; i<m_totalNo; i++){
			TPTable[(int) m_rawStat[i][0]][(int) m_rawStat[i][1]]++;
			m_prfStat[i] = calculateCurrentPRF(TPTable);
		}
	}
	
	public double[][] calculateCurrentPRF(int[][] TPTable){
		double[][] prf = new double[TPTable.length][3];//column: 0-1 class; row: precision, recall, F1.
		for(int i=0; i<TPTable.length; i++){
			prf[i][0] = TPTable[i][i]/Utils.sumOfRow(TPTable, i);// Precision.
			prf[i][1] = TPTable[i][i]/Utils.sumOfColumn(TPTable, i); // Recall.
			prf[i][2] = 2*prf[i][0]*prf[i][1]/(prf[i][0] + prf[i][1]); // F1.
		}
		return prf;
	}
}
