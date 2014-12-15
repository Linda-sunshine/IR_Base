package Analyzer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import structures._stat;
import utils.Utils;

public class FeatureSelector {

	double m_startProb;
	double m_endProb;
	ArrayList<String> m_selectedFeatures;

	//Default setting of feature selection
	public FeatureSelector(){
		m_startProb = 0;
		m_endProb = 1;
	}
		
	//Given start and end of feature selection.
	public FeatureSelector(double startProb, double endProb){
		m_startProb = startProb;
		m_endProb = endProb;
	}
	
	//Return the selected features.
	public ArrayList<String> getSelectedFeatures(){
		return m_selectedFeatures;
	}
	
	//Sort the features according to integer values.
	public void sortFeaturesInt(HashMap<Integer, ArrayList<String>> featureValues){
		ArrayList<String> sortedFeatures = new ArrayList<String>();
		Collection<Integer> unsortedValues = featureValues.keySet();
		ArrayList<Integer> sortedValues = new ArrayList<Integer>(unsortedValues);     
		Collections.sort(sortedValues);//sort the DF
		for(int value : sortedValues) sortedFeatures.addAll(featureValues.get(value));//sort the features according to sorted DF/IG/MI/CHI.
			
		//Start fetching particular features.
		int totalSize = sortedFeatures.size();
		int start = (int) (totalSize * m_startProb);
		int end = (int) (totalSize * m_endProb);
		m_selectedFeatures = new ArrayList<String>(sortedFeatures.subList(start, end));
	}
	
	//Sort the features according to double values.
	public void sortFeaturesDouble(HashMap<Double, ArrayList<String>> featureValues){
		ArrayList<String> sortedFeatures = new ArrayList<String>();
		Collection<Double> unsortedValues = featureValues.keySet();
		ArrayList<Double> sortedValues = new ArrayList<Double>(unsortedValues);     
		Collections.sort(sortedValues);//sort the DF
		for(double value : sortedValues) sortedFeatures.addAll(featureValues.get(value));//sort the features according to sorted DF/IG/MI/CHI.
			
		//Start fetching particular features.
		int totalSize = sortedFeatures.size();
		int start = (int) (totalSize * m_startProb);
		int end = (int) (totalSize * m_endProb);
		m_selectedFeatures = new ArrayList<String>(sortedFeatures.subList(start, end));
	}
	
	//Feature Selection -- DF.
	public void DF(HashMap<String, _stat> featureStat, int threshold){
		HashMap<Integer, ArrayList<String>> featureDF = new HashMap<Integer, ArrayList<String>>();
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if(sumDF > threshold){
				if(featureDF.containsKey(sumDF)){
					featureDF.get(sumDF).add(f);
				}else{
					ArrayList<String> temp = new ArrayList<String>();
					temp.add(f);
					featureDF.put(sumDF, temp);
				}
			}
		}
		sortFeaturesInt(featureDF);
	}
		
	//Feature Selection -- IG.
	public void IG(HashMap<String, _stat> featureStat, int[] classMemberNo, int threshold){
		HashMap<Double, ArrayList<String>> featureIG = new HashMap<Double, ArrayList<String>>();
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
			if (sumDF > threshold){
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
				if(featureIG.containsKey(Gt)){
					featureIG.get(Gt).add(f);
				}else{
					ArrayList<String> tempArray = new ArrayList<String>();
					tempArray.add(f);
					featureIG.put(Gt, tempArray);
				}
			}
		}
		sortFeaturesDouble(featureIG);
	} 
		
	//Feature Selection -- MI.
	public void MI(HashMap<String, _stat> featureStat, int[] classMemberNo, int threshold){
		HashMap<Double, ArrayList<String>> featureMI = new HashMap<Double, ArrayList<String>>();
		double[] PrCi = new double[classMemberNo.length];
		double[] ItCi = new double[classMemberNo.length];
		double N = Utils.sumOfArray(classMemberNo);
		double Iavg = 0;

		for (int i = 0; i < classMemberNo.length; i++)
			PrCi[i] = classMemberNo[i] / N;
		for (String f : featureStat.keySet()) {
			// Filter the features which have smaller DFs.
			int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if (sumDF > threshold) {
				Iavg = 0;
				for (int i = 0; i < classMemberNo.length; i++) {
					_stat temp = featureStat.get(f);
					double A = temp.getDF()[i];
					ItCi[i] = Math.log(A * N / classMemberNo[i]
							* Utils.sumOfArray(temp.getDF()));
					Iavg += ItCi[i] * PrCi[i];
				}
				if (featureMI.containsKey(Iavg)) {
					featureMI.get(Iavg).add(f);
				} else {
					ArrayList<String> tempArray = new ArrayList<String>();
					tempArray.add(f);
					featureMI.put(Iavg, tempArray);
				}
			}
		}
		sortFeaturesDouble(featureMI);
	}
		
	//Feature Selection -- CHI.
	public void CHI(HashMap<String, _stat> featureStat, int[] classMemberNo, int threshold){
		int classNo = classMemberNo.length;
		HashMap<Double, ArrayList<String>> featureCHI = new HashMap<Double, ArrayList<String>>();
		double N = Utils.sumOfArray(classMemberNo);
		double[] PrCi = new double [classNo]; 
		double[] X2tc = new double [classNo];
		double X2avg = 0;
		for(int i = 0; i < classNo; i++) PrCi[i] = classMemberNo[i] / N;//Get the class probability.
			
		for(String f: featureStat.keySet()){
			//Filter the features which have smaller DFs.
			int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if (sumDF > threshold){
				X2avg = 0;
				for(int i = 0; i < classNo; i++){
					_stat temp = featureStat.get(f);
					double A = temp.getDF()[i];
					double B = Utils.sumOfArray(temp.getDF()) - A;
					double C = classMemberNo[i] - A;
					double D = Utils.sumOfArray(temp.getCounts()[1]) - C;
					X2tc[i] = N * ( A * D - B * C ) * ( A * D - B * C ) / ( A + C ) * ( B + D ) * ( A + B ) * ( C + D );
					X2avg += Math.pow(X2tc[i], PrCi[i]);
				}
				//X2max = Utils.maxOfArrayValue(X2tc);
				if(featureCHI.containsKey(X2avg)){
					featureCHI.get(X2avg).add(f);
				} else{
					ArrayList<String> tempArray = new ArrayList<String>();
					tempArray.add(f);
					featureCHI.put(X2avg, tempArray);
				}
			}
		}
		sortFeaturesDouble(featureCHI);
	}
}