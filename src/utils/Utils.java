package utils;

import structures._SparseFeature;

public class Utils {
	
	//Find the max value's index of an array, return Index of the maximum.
	public static int maxOfArrayIndex(double[] probs){
		int maxIndex = 0;
		double maxValue = probs[0];
		for(int i = 0; i < probs.length; i++){
			if(probs[i] > maxValue){
				maxValue = probs[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	//Find the max value's index of an array, return Value of the maximum.
	public static double maxOfArrayValue(double[] probs){
		double maxValue = probs[0];
		for(int i = 0; i < probs.length; i++){
			if(probs[i] > maxValue){
				maxValue = probs[i];
			}
		}
		return maxValue;
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
			int index = sf[i].getIndex() + 1;
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
}
