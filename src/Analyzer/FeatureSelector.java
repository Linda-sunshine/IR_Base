package Analyzer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import structures._RankItem;
import structures._stat;
import utils.Utils;

/**
 * 
 * @author Lin Gong
 * Implementation of several text feature selection algorithms.
 * Yang, Yiming, and Jan O. Pedersen. "A comparative study on feature selection in text categorization." ICML. Vol. 97. 1997.
 */
public class FeatureSelector {

	double m_startProb, m_endProb; // selecting features proportionally
	int m_maxDF, m_minDF; // upper and lower bounds for DF in feature selection (exclusive!)
	
	ArrayList<_RankItem> m_selectedFeatures;

	//Default setting of feature selection
	public FeatureSelector(){
		m_startProb = 0;
		m_endProb = 1;
		m_selectedFeatures = new ArrayList<_RankItem>();
	}
		
	//Given start and end of feature selection.
	public FeatureSelector(double startProb, double endProb, int maxDF, int minDF){
		if (startProb>endProb) {
			double t = startProb;
			startProb = endProb;
			endProb = t;
		}
		
		m_startProb = Math.max(0, startProb);
		m_endProb = Math.min(1.0, endProb);
		m_maxDF = maxDF<=0 ? Integer.MAX_VALUE : maxDF;
		m_minDF = minDF; 
		m_selectedFeatures = new ArrayList<_RankItem>();
	}
	
	//Return the selected features.
	public ArrayList<String> getSelectedFeatures(){
		ArrayList<String> features = new ArrayList<String>();
		Collections.sort(m_selectedFeatures);//ascending order
		
		int totalSize = m_selectedFeatures.size();
		
		System.out.println("total\t"+totalSize);
		//for ArsTech
//		int totalCorpusSize = 4000;
//		// int start = (int) (totalSize * m_startProb);
//		// int end = (int) (totalSize * m_endProb);
//		//
//		// for(int i=start; i<end; i++)
//		// features.add(m_selectedFeatures.get(i).m_name);
//		
//		int upDF = (int) (totalCorpusSize * 0.3);
//		int bottomDF = (int) (totalCorpusSize * 0.00001);

//		System.out.println("up DF\t" + upDF + "\t bottom DF \t" + bottomDF);

//		for (int i = 0; i < totalSize; i++) {
//			if ((m_selectedFeatures.get(i).m_value < upDF)
//					&& (m_selectedFeatures.get(i).m_value > bottomDF)) {
//				features.add(m_selectedFeatures.get(i).m_name);
//			}
//		}
		
		int start = (int) (totalSize * m_startProb);
		int end = (int) (totalSize * m_endProb);
		System.out.println("start feature val\t"+m_selectedFeatures.get(start).m_value+"\t"+"end feature val\t"+m_selectedFeatures.get(end).m_value);
		for(int i=start; i<end; i++)
			features.add(m_selectedFeatures.get(i).m_name);
		
		return features;
	}
	
	//Feature Selection -- DF.
	public void DF(HashMap<String, _stat> featureStat){
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			double sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if(sumDF > m_minDF && sumDF < m_maxDF)
				m_selectedFeatures.add(new _RankItem(f, sumDF));
		}
		
		System.out.println("selected features size\t"
				+ m_selectedFeatures.size()+"\t maxDF\t"+m_maxDF+"\t minDF\t"+m_minDF);
		
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
			if (sumDF > m_minDF && sumDF < m_maxDF){
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
			if (sumDF > m_minDF && sumDF < m_maxDF) {
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
		int N = Utils.sumOfArray(classMemberNo), sumDF;
		double[] X2tc = new double [classNo];
		double X2avg = 0;		
			
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			_stat temp = featureStat.get(f);
			sumDF = Utils.sumOfArray(temp.getDF());
			
			if (sumDF > m_minDF && sumDF < m_maxDF) {	
				X2avg = 0;				
				for(int i = 0; i < classNo; i++){
					X2tc[i] = Utils.ChiSquare(N, sumDF, temp.getDF()[i], classMemberNo[i]);
					X2avg += X2tc[i] * classMemberNo[i] / N;
				}
				//X2max = Utils.maxOfArrayValue(X2tc);
				m_selectedFeatures.add(new _RankItem(f, X2avg));
			}
		}
	}
}