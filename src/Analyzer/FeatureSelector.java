package Analyzer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import structures._stat;
import utils.Utils;

public class FeatureSelector {

	double m_startProb;
	double m_endProb;

	//Default setting of feature selection
	public FeatureSelector(){
		this.m_startProb = 0;
		this.m_endProb = 1;
	}
		
	//Given start and end of feature selection.
	public FeatureSelector(double startProb, double endProb){
		this.m_startProb = startProb;
		this.m_endProb = endProb;
	}
		
	//Feature Selection -- DF.
	public ArrayList<String> DF(HashMap<String, _stat> featureStat, int threshold){
		HashMap<Integer, ArrayList<String>> featureDF = new HashMap<Integer, ArrayList<String>>();
		ArrayList<String> sortedFeatures = new ArrayList<String>();
		Collection<Integer> unsortedDF = new HashSet<Integer>();
			
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
		unsortedDF = featureDF.keySet();
		ArrayList<Integer> sortedDF = new ArrayList<Integer>(unsortedDF);     
		Collections.sort(sortedDF);//sort the DF
		for(int DF : sortedDF) sortedFeatures.addAll(featureDF.get(DF));//sort the features according to sorted DF.
			
		//Start fetching particular features. How do we define the criterion?
		int totalSize = featureStat.size();
		int start = (int) (totalSize * this.m_startProb);
		int end = (int) (totalSize * this.m_endProb);
		ArrayList<String> selectedFeatures = new ArrayList<String>(sortedFeatures.subList(start, end));
		return selectedFeatures;
	}
		
	//Feature Selection -- IG.
	public ArrayList<String> IG(HashMap<String, _stat> featureStat, int[] classMemberNo, int threshold){
		HashMap<Double, ArrayList<String>> featureIG = new HashMap<Double, ArrayList<String>>();
		ArrayList<String> sortedFeatures = new ArrayList<String>();
		Collection<Double> unsortedIG = new HashSet<Double>();
			
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
		int featureSize2 = 0; // Use this to filter features' DF < threshold.
		for(String f: featureStat.keySet()){
			
			//Filter the features which have smaller DFs.
			int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
			if (sumDF > threshold){
				featureSize2++;
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
		unsortedIG = featureIG.keySet();
		ArrayList<Double> sortedIG = new ArrayList<Double>(unsortedIG);     
		Collections.sort(sortedIG);//sort the DF
		for(double IG : sortedIG) sortedFeatures.addAll(featureIG.get(IG));//sort the features according to sorted DF.
			
		int start = (int) (featureSize2 * this.m_startProb);
		int end = (int) (featureSize2 * this.m_endProb);
		ArrayList<String> selectedFeatures = new ArrayList<String>(sortedFeatures.subList(start, end));
		return selectedFeatures;
	} 
		
	//Feature Selection -- MI.
	public ArrayList<String> MI(HashMap<String, _stat> featureStat, int[] classMemberNo, int threshold){
			HashMap<Double, ArrayList<String>> featureMI = new HashMap<Double, ArrayList<String>>();
			ArrayList<String> sortedFeatures = new ArrayList<String>();
			Collection<Double> unsortedMI = new HashSet<Double>();
			
			double[] PrCi = new double [classMemberNo.length]; 
			double[] ItCi = new double [classMemberNo.length];
			double N = Utils.sumOfArray(classMemberNo);
			double Iavg = 0;
			
			for(int i = 0; i < classMemberNo.length; i++) PrCi[i] = classMemberNo[i] / N;
			for(String f: featureStat.keySet()){
				//Filter the features which have smaller DFs.
				int sumDF = Utils.sumOfArray(featureStat.get(f).getDF());
				if (sumDF > threshold){
					Iavg = 0;
					for(int i = 0; i < classMemberNo.length; i++){
						_stat temp = featureStat.get(f);
						double A = temp.getDF()[i];
						ItCi[i] = Math.log(A * N / classMemberNo[i] * Utils.sumOfArray(temp.getDF()));
						Iavg += ItCi[i] * PrCi[i];				
					}
					if(featureMI.containsKey(Iavg)){
						featureMI.get(Iavg).add(f);
					}else{
						ArrayList<String> tempArray = new ArrayList<String>();
						tempArray.add(f);
						featureMI.put(Iavg, tempArray);
					}
				}else System.out.println("No features found as for threshold " + threshold);
			}
				
			unsortedMI = featureMI.keySet();
			ArrayList<Double> sortedMI = new ArrayList<Double>(unsortedMI);     
			Collections.sort(sortedMI);//sort the DF
			for(double MI : sortedMI) sortedFeatures.addAll(featureMI.get(MI));//sort the features according to sorted MI.
			
			//Start fetching particular features. How do we define the criterion?
			int totalSize = featureStat.size();
			int start = (int) (totalSize * this.m_startProb);
			int end = (int) (totalSize * this.m_endProb);
			ArrayList<String> selectedFeatures = new ArrayList<String>(sortedFeatures.subList(start, end));
			return selectedFeatures;
		}
		
	//Feature Selection -- CHI.
	public ArrayList<String> CHI(HashMap<String, _stat> featureStat, int[] classMemberNo, int threshold){
		int classNo = classMemberNo.length;
		HashMap<Double, ArrayList<String>> featureCHI = new HashMap<Double, ArrayList<String>>();
		ArrayList<String> sortedFeatures = new ArrayList<String>();
		Collection<Double> unsortedCHI = new HashSet<Double>();
		
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
			} else System.out.println("No features found as for threshold " + threshold);
		}
		unsortedCHI = featureCHI.keySet();
		ArrayList<Double> sortedCHI = new ArrayList<Double>(unsortedCHI);     
		Collections.sort(sortedCHI);//sort the DF
		for(double CHI : sortedCHI) sortedFeatures.addAll(featureCHI.get(CHI));//sort the features according to sorted MI.
			
		//Start fetching particular features. How do we define the criterion?
		int totalSize = featureStat.size();
		int start = (int) (totalSize * this.m_startProb);
		int end = (int) (totalSize * this.m_endProb);
		ArrayList<String> selectedFeatures = new ArrayList<String>(sortedFeatures.subList(start, end));
		return selectedFeatures;
	}
}