package utils;

import structures._Doc;
import structures._SparseFeature;

public class Utils {
	
	//Find the max value's index of an array, return Index of the maximum.
	public static int maxOfArrayIndex(double[] probs){
		int maxIndex = 0;
		double maxValue = probs[0];
		for(int i = 1; i < probs.length; i++){
			if(probs[i] > maxValue){
				maxValue = probs[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	//Find the max value's index of an array, return Value of the maximum.
	public static double maxOfArrayValue(double[] probs){
		return probs[maxOfArrayIndex(probs)];
	}
	
	//This function is used to calculate the log of the sum of several values.
	public static double logSumOfExponentials(double[] xs){
		double max = maxOfArrayValue(xs);
		double sum = 0.0;
		if(xs.length == 1){
			return xs[0];
		}
		for (int i = 0; i < xs.length; i++) {
			if (xs[i] != Double.NEGATIVE_INFINITY) {
				sum += Math.exp(xs[i] - max);
			}
		}
		return Math.log(sum) + max;
	}
	
	//The function is used to calculate the sum of log of two arrays.
	public static double sumLog(double[] probs, double[] values){
		double result = 0;
		if(probs.length == values.length){
			for(int i = 0; i < probs.length; i++){
				result += values[i] * Math.log(probs[i]);
			}
		} else{
			System.out.println("log sum fails due to the lenghts of two arrars are not matched!!");
		}
		return result;
	}
	
	//The function defines the dot product of beta and sparse Vector of a document.
	public static double dotProduct(double[] beta, _SparseFeature[] sf){
		double sum = beta[0];
		for(int i = 0; i < sf.length; i++){
			int index = sf[i].getIndex() + 1;//this design is very risky, you'd better double check the usage of this function!!
			sum += beta[index] * sf[i].getNormValue();
		}
		return sum;
	}
		
	//The function defines the dot product of two normal arrays.
	public static double dotProduct(double[] a, double[] b){
		double result = 0.0;
		if(a.length != b.length){
			System.out.println("dotProduct error! The length of two arrays is not matched!");
		} else{
			for(int i = 0; i < a.length; i++){
				result += a[i] * b[i];
			}
		}
		return result;
	}
	
	//The function defines the sum of an array.
	public static int sumOfArray(int[] a){
		int sum = 0;
		for (int i: a)
			sum += i;
		return sum;
	}
	
	//The function defines the sum of an array.
	public static double sumOfArray(double[] a) {
		double sum = 0;
		for (double i : a)
			sum += i;
		return sum;
	}
	
	//L1 normalization.
	//Calculate the sum of all the values of features.
	static public double sumOfFeatures(_SparseFeature[] fs) {
		double sum = 0;
		for (_SparseFeature feature: fs)
			sum += Math.abs(feature.getValue());
		return sum;
	}
	
	//Set the normalized value back to the sparse feature.
	static public void L1Normalization(_SparseFeature[] fs) {
		double sum = sumOfFeatures(fs);
		if (sum>0) {
			//L1 length normalization
			for(_SparseFeature f:fs){
				double normValue = f.getValue()/sum;
				f.setNormValue(normValue);
			}
		} else{
			for(_SparseFeature f: fs){
				f.setNormValue(0.0);
			}
		}
	}
	
	//L2 normalization.
	
	//L2 = sqrt(sum of fsValue*fsValue).
	static public double sumOfFeaturesL2(_SparseFeature[] fs) {
		double sum = 0;
		for (_SparseFeature feature: fs){
			double value = feature.getValue();
			sum += value * value;
		}
		return Math.sqrt(sum);
	}
	
	static public void L2Normalization(_SparseFeature[] fs) {
		double sum = sumOfFeaturesL2(fs);
		if (sum>0) {
			//L1 length normalization
			for(_SparseFeature f: fs){
				double normValue = f.getValue()/sum;
				f.setNormValue(normValue);
			}
		}
		else{
			for(_SparseFeature f: fs){
				f.setNormValue(0.0);
			}
		}
	}
	
	//Calculate the similarity between two documents.
	public static double calculateSimilarity(_Doc d1, _Doc d2){
		return calculateSimilarity(d1.getSparse(), d2.getSparse());
	}
	
	//Calculate the similarity between two sparse vectors.
	public static double calculateSimilarity(_SparseFeature[] spVct1, _SparseFeature[] spVct2) {
		double similarity = 0;
		int pointer1 = 0, pointer2 = 0;
		while (pointer1 < spVct1.length && pointer2 < spVct2.length) {
			_SparseFeature temp1 = spVct1[pointer1];
			_SparseFeature temp2 = spVct2[pointer2];
			if (temp1.getIndex() == temp2.getIndex()) {
				similarity += temp1.getValue() * temp2.getValue();
				pointer1++;
				pointer2++;
			} else if (temp1.getIndex() > temp2.getIndex())
				pointer2++;
			else
				pointer1++;
		}
		return similarity;
	}
		
	public static void main(String[] args){
		_SparseFeature s1 = new _SparseFeature(1, 4);
		_SparseFeature s2 = new _SparseFeature(5, 1);
		_SparseFeature s3 = new _SparseFeature(6, 8);
		_SparseFeature s4 = new _SparseFeature(9, 4);
		_SparseFeature s5 = new _SparseFeature(1, 3);
		_SparseFeature s6 = new _SparseFeature(2, 4);
		_SparseFeature s7 = new _SparseFeature(5, 1);
		_SparseFeature s8 = new _SparseFeature(9, 2);

		_SparseFeature[] d1 = new _SparseFeature[] {s1, s2, s3, s4};
		_SparseFeature[] d2 = new _SparseFeature[] {s5, s6, s7, s8};
		double similarity = calculateSimilarity(d1, d2);
		System.out.println(similarity);

		
	}
}
