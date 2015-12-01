package structures;

import java.util.ArrayList;

import utils.Utils;

public class _PerformanceStat {
	//Parameters in batch mode.
	int[][] m_TPTable;
	double[][] m_onePRFStat; //This is the performance used in batch mode.
	
	//Parameters in online mode.
	int m_totalNo;
	int[][] m_rawStat;//This stat stores the raw data of predicted reviews.
	
	//Used in CoLinAdapt for storing results.
	ArrayList<Integer> m_rawStatTrueL;
	ArrayList<Integer> m_rawStatPredL;
	
	double[][][] m_PRFStat; // Store the performance for each new prediction result.
	
	public _PerformanceStat(int predL, int trueL){
		m_rawStatTrueL = new ArrayList<Integer>();
		m_rawStatTrueL.add(trueL);
		m_rawStatPredL = new ArrayList<Integer>();
		m_rawStatPredL.add(predL);
	}
	//Constructor for batch mode.
	public _PerformanceStat(int[][] TPTable){
		m_TPTable = TPTable;
	}
	
	//Constructor for online mode.
	public _PerformanceStat(int[] predLs, int[] trueLs){
		m_totalNo = trueLs.length;
		
		m_rawStat = new int[2][m_totalNo];
		m_rawStat[0] = predLs;
		m_rawStat[1] = trueLs;
		
		m_TPTable = new int[2][2];
	}
	
	//Add one pred result to the stat.
	public void addOnePredResult(int predL, int trueL){
		m_rawStatTrueL.add(trueL);
		m_rawStatPredL.add(predL);
	}
	//In online mode, prf for different reviews.
	public void calcuatePreRecF1(){
		m_PRFStat = new double[m_totalNo][][];
		for(int i=0; i<m_totalNo; i++){
			m_TPTable[m_rawStat[0][i]][m_rawStat[1][i]]++;
			m_PRFStat[i] = calculateCurrentPRF();
		}
	}
	
	public double[][] calculateCurrentPRF(){
		double[][] prf = new double[m_TPTable.length][3];//column: 0-1 class; row: precision, recall, F1.
		for(int i=0; i<m_TPTable.length; i++){
			prf[i][0] = m_TPTable[i][i]/(Utils.sumOfRow(m_TPTable, i) + 0.00001);// Precision.
			prf[i][1] = m_TPTable[i][i]/(Utils.sumOfColumn(m_TPTable, i) + 0.00001); // Recall.
			prf[i][2] = 2*prf[i][0]*prf[i][1]/(prf[i][0] + prf[i][1] + 0.00001); // F1.
		}
		return prf;
	}
	//In batch mode, one prf for each user.
	public void setOnePRFStat(double[][] prf){
		m_onePRFStat = prf;
	}
}
