package Analyzer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import structures._RankItem;
import structures._stat;
import utils.Utils;

public class FeatureSelector {

	double m_startProb;
	double m_endProb;
	int m_DFThreshold;
	ArrayList<_RankItem> m_selectedFeatures;

	//Default setting of feature selection
	public FeatureSelector(){
		m_startProb = 0;
		m_endProb = 1;
		m_selectedFeatures = new ArrayList<_RankItem>();
	}
		
	//Given start and end of feature selection.
	public FeatureSelector(double startProb, double endProb, int DFThreshold){
		if (startProb>endProb) {
			double t = startProb;
			startProb = endProb;
			endProb = t;
		}
		
		m_startProb = Math.max(0, startProb);
		m_endProb = Math.min(1.0, endProb);
		m_DFThreshold = DFThreshold;
		m_selectedFeatures = new ArrayList<_RankItem>();
	}
	
	//Return the selected features.
	public ArrayList<String> getSelectedFeatures(){
		ArrayList<String> features = new ArrayList<String>();
		Collections.sort(m_selectedFeatures);//ascending order
		
		int totalSize = m_selectedFeatures.size();
		int start = (int) (totalSize * m_startProb);
		int end = (int) (totalSize * m_endProb);
		for(int i=start; i<end; i++)
			features.add(m_selectedFeatures.get(i).m_name);
		
		return features;
	}
	
	//Feature Selection -- DF.
	public void DF(HashMap<String, _stat> featureStat){
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			double sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if(sumDF > m_DFThreshold){
				m_selectedFeatures.add(new _RankItem(f, sumDF));
			}
		}
	}
		
	//Feature Selection -- IG.
	public void IG(HashMap<String, _stat> featureStat, int[] classMemberNo){
		double classMemberSum = Utils.sumOfArray(classMemberNo);
		double[] PrCi = new double [classMemberNo.length];//I
		double[] PrCit = new double [classMemberNo.length];//II
		double[] PrCitNot = new double [classMemberNo.length];//III
			
		double Prt = 0, PrtNot = 0;
		double Gt = 0;//IG
		double PrCiSum = 0, PrCitSum = 0, PrCitNotSum = 0;
			
		//- $sigma$PrCi * log PrCi
		for(int i = 0; i < classMemberNo.length; i++) {
			PrCi[i] = classMemberNo[i] / classMemberSum;
			if(PrCi[i] != 0){
				PrCiSum -= PrCi[i] * Math.log(PrCi[i]);
			}
		}
		
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if (sumDF > m_DFThreshold){
				_stat temp = featureStat.get(f);
				Prt = Utils.sumOfArray(temp.getDF()) / classMemberSum;
				PrtNot = 1 - Prt;
				PrCitSum = 0;
				PrCitNotSum = 0;
				for(int i = 0; i < classMemberNo.length; i++){
					PrCit[i] = ((double)temp.getDF()[i] / classMemberNo[i]) * PrCi[i] / Prt;
					PrCitNot[i] = ((double)(classMemberNo[i] - temp.getDF()[i]) / classMemberNo[i]) * PrCi[i] / PrtNot;
					if(PrCit[i] != 0){
						PrCitSum += PrCit[i] * Math.log(PrCit[i]);
					}
					if(PrCitNot[i] != 0){
						PrCitNotSum += PrCitNot[i] * Math.log(PrCi[i]);
					}
				}
				Gt = PrCiSum + PrCitSum + PrCitNotSum;
				m_selectedFeatures.add(new _RankItem(f, Gt));
			}
		}
	} 
		
	//Feature Selection -- MI.
	public void MI(HashMap<String, _stat> featureStat, int[] classMemberNo){
		double[] PrCi = new double[classMemberNo.length];
		double[] ItCi = new double[classMemberNo.length];
		double N = Utils.sumOfArray(classMemberNo);
		double Iavg = 0;

		for (int i = 0; i < classMemberNo.length; i++)
			PrCi[i] = classMemberNo[i] / N;
		for (String f : featureStat.keySet()) {
			// Filter the features which have smaller DFs.
			int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if (sumDF > m_DFThreshold) {
				Iavg = 0;
				for (int i = 0; i < classMemberNo.length; i++) {
					_stat temp = featureStat.get(f);
					double A = temp.getDF()[i];
					ItCi[i] = Math.log(A * N / classMemberNo[i]
							* Utils.sumOfArray(temp.getDF()));
					Iavg += ItCi[i] * PrCi[i];
				}
				
				m_selectedFeatures.add(new _RankItem(f, Iavg));
			}
		}
	}
		
	//Feature Selection -- CHI.
	public void CHI(HashMap<String, _stat> featureStat, int[] classMemberNo){
		int classNo = classMemberNo.length;
		int N = Utils.sumOfArray(classMemberNo), DF;
		double[] X2tc = new double [classNo];
		double X2avg = 0;		
			
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			_stat temp = featureStat.get(f);
			DF = Utils.sumOfArray(temp.getDF());
			
			if (DF > m_DFThreshold){				
				X2avg = 0;				
				for(int i = 0; i < classNo; i++){
					X2tc[i] = ChiSquare(N, DF, temp.getDF()[i], classMemberNo[i]);
					X2avg += X2tc[i] * classMemberNo[i] / N;
				}
				//X2max = Utils.maxOfArrayValue(X2tc);
				m_selectedFeatures.add(new _RankItem(f, X2avg));
			}
		}
	}
	
	/**
	 * 
	 * @param N: total document size
	 * @param DF: document frequency for term t
	 * @param tcDF: number of documents where t and c co-occur
	 * @param cDF: number of documents where t occurs
	 * @return
	 */
	static public double ChiSquare(int N, int DF, int tcDF, int cDF) {
		double A = tcDF;//t & c
		double B = DF - A;//t & !c
		double C = cDF - A;//!t & c
		double D = N - DF - cDF + A;//!t & !c
		
		return N * ( A * D - B * C ) * ( A * D - B * C ) / cDF / ( B + D ) / DF / ( C + D );
	}
}