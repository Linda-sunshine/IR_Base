package structures;

import utils.Utils;

public class _PerformanceStat {
	//Parameters in batch mode.
	int[][] m_TPTable;
	double[][] m_onePRFStat; //This is the performance used in batch mode.
	
	//Parameters in online mode.
	int m_totalNo;
	int[][] m_rawStat;//This stat stores the raw data of predicted reviews.
	double[][][] m_PRFStat; // Store the performance for each new prediction result.
	
	//Constructor for batch mode.
	public _PerformanceStat(int[][] TPTable){
		m_TPTable = TPTable;
	}
	
	//Constructor for online mode.
	public _PerformanceStat(int[] trueLs, int[] predLs){
		m_totalNo = trueLs.length;
		
		m_rawStat = new int[2][m_totalNo];
		m_rawStat[0] = trueLs;
		m_rawStat[1] = predLs;
		
		m_TPTable = new int[2][2];
	}
	//In online mode, prf for different reviews.
	public void calcuatePreRecF1(){
		m_PRFStat = new double[m_totalNo][][];
		for(int i=0; i<m_totalNo; i++){
			m_TPTable[m_rawStat[i][0]][m_rawStat[i][1]]++;
			m_PRFStat[i] = calculateCurrentPRF();
		}
	}
	
	public double[][] calculateCurrentPRF(){
		double[][] prf = new double[m_TPTable.length][3];//column: 0-1 class; row: precision, recall, F1.
		for(int i=0; i<m_TPTable.length; i++){
			prf[i][0] = m_TPTable[i][i]/Utils.sumOfRow(m_TPTable, i);// Precision.
			prf[i][1] = m_TPTable[i][i]/Utils.sumOfColumn(m_TPTable, i); // Recall.
			prf[i][2] = 2*prf[i][0]*prf[i][1]/(prf[i][0] + prf[i][1]); // F1.
		}
		return prf;
	}
	//In batch mode, one prf for each user.
	public void setOnePRFStat(double[][] prf){
		m_onePRFStat = prf;
	}
}
