package utils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import json.JSONException;
import json.JSONObject;
import structures._Doc;
import structures._SparseFeature;

public class Utils {
	
	//Find the max value's index of an array, return Index of the maximum.
	public static int maxOfArrayIndex(double[] probs){
		return maxOfArrayIndex(probs, probs.length);
	}
	
	public static int maxOfArrayIndex(double[] probs, int length){
		int maxIndex = 0;
		double maxValue = probs[0];
		for(int i = 1; i < length; i++){
			if(probs[i] > maxValue){
				maxValue = probs[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int minOfArrayIndex(double[] probs){
		return minOfArrayIndex(probs, probs.length);
	}
	
	public static int minOfArrayIndex(double[] probs, int length){
		int minIndex = 0;
		double minValue = probs[0];
		for(int i = 1; i < length; i++){
			if(probs[i] < minValue){
				minValue = probs[i];
				minIndex = i;
			}
		}
		return minIndex;
	}
	
	//Calculate the sum of a column in an array.
	public static double sumOfRow(double[][] mat, int i){
		return sumOfArray(mat[i]);
	}
	
	//Calculate the sum of a row in an array.
	public static double sumOfColumn(double[][] mat, int i){
		double sum = 0;
		for(int j = 0; j < mat.length; j++){
			sum += mat[j][i];
		}
		return sum;
	}
	
	//Calculate the sum of a column in an array.
	public static int sumOfRow(int[][] mat, int i){
		return sumOfArray(mat[i]);
	}
	
	//Calculate the sum of a row in an array.
	public static int sumOfColumn(int[][] mat, int i){
		int sum = 0;
		for(int j = 0; j < mat.length; j++){
			sum += mat[j][i];
		}
		return sum;
	}
	
	public static double entropy(double[] prob, boolean logScale) {
		double ent = 0;
		for(double p:prob) {
			if (logScale)
				ent += Math.exp(p) * p;
			else
				ent += Math.log(p) * p;
		}
		return -ent;
	}
	
	//Find the max value's index of an array, return Value of the maximum.
	public static double maxOfArrayValue(double[] probs){
		return probs[maxOfArrayIndex(probs)];
	}
	
	//This function is used to calculate the log of the sum of several values.
	public static double logSumOfExponentials(double[] xs){
		if(xs.length == 1){
			return xs[0];
		}
		
		double max = maxOfArrayValue(xs);
		double sum = 0.0;
		for (int i = 0; i < xs.length; i++) {
			if (!Double.isInfinite(xs[i])) 
				sum += Math.exp(xs[i] - max);
		}
		return Math.log(sum) + max;
	}
	
	public static double logSum(double log_a, double log_b)
	{
		if (log_a < log_b)
			return log_b+Math.log(1 + Math.exp(log_a-log_b));
		else
			return log_a+Math.log(1 + Math.exp(log_b-log_a));
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
	public static double dotProduct(double[] beta, _SparseFeature[] sf, int offset){
		double sum = beta[offset];
		for(int i = 0; i < sf.length; i++){
			int index = sf[i].getIndex() + offset + 1;
			sum += beta[index] * sf[i].getValue();
		}
		return sum;
	}
	
	public static double dotProduct(double[] a, double[] b) {
		if (a.length != b.length)
			return Double.NaN;
		double sum = 0;
		for(int i=0; i<a.length; i++)
			sum += a[i] * b[i];
		return sum;
	}
	
	public static double L2Norm(double[] a) {
		return Math.sqrt(dotProduct(a,a));
	}
	
	//Logistic function: 1.0 / (1.0 + exp(-wf))
	public static double logistic(double[] fv, double[] w){
		double sum = w[0];//start from bias term
		for(int i = 0; i < fv.length; i++)
			sum += fv[i] * w[1+i];
		return 1.0 / (1.0 + Math.exp(-sum));
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
	
	//The function defines the sum of an array.
	public static void scaleArray(double[] a, double b) {
		for (int i=0; i<a.length; i++)
			a[i] *= b;
	}
	
	//The function defines the sum of an array.
	public static void scaleArray(double[] a, double[] b, double scale) {
		for (int i=0; i<a.length; i++)
			a[i] += b[i] * scale;
	}
	
	//The function defines the sum of an array.
	public static void setArray(double[] a, double[] b, double scale) {
		for (int i=0; i<a.length; i++)
			a[i] = b[i] * scale;
	}
	
	//L1 normalization: fsValue/sum(abs(fsValue))
	static public double sumOfFeaturesL1(_SparseFeature[] fs) {
		double sum = 0;
		for (_SparseFeature feature: fs)
			sum += Math.abs(feature.getValue());
		return sum;
	}
	
	//Set the normalized value back to the sparse feature.
	static public void L1Normalization(_SparseFeature[] fs) {
		double sum = sumOfFeaturesL1(fs);
		if (sum>0) {
			//L1 length normalization
			for(_SparseFeature f:fs){
				double normValue = f.getValue()/sum;
				f.setValue(normValue);
			}
		} else{
			for(_SparseFeature f: fs){
				f.setValue(0.0);
			}
		}
	}
	
	//L2 normalization: fsValue/sqrt(sum of fsValue*fsValue)
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
			for(_SparseFeature f: fs){
				double normValue = f.getValue()/sum;
				f.setValue(normValue);
			}
		}
		else{
			for(_SparseFeature f: fs){
				f.setValue(0.0);
			}
		}
	}
	
	static public String getJSONValue(JSONObject json, String key) {
		try {
			if (json.has(key))				
				return(json.getString(key));
			else
				return "NULL";
		} catch (JSONException e) {
			return "NULL";
		}
	}
	
	//Calculate the similarity between two documents.
	public static double calculateSimilarity(_Doc d1, _Doc d2){
		return calculateSimilarity(d1.getSparse(), d2.getSparse());
	}
	
	public static double cosine(_SparseFeature[] spVct1, _SparseFeature[] spVct2) {
		return calculateSimilarity(spVct1, spVct2) / sumOfFeaturesL2(spVct1) / sumOfFeaturesL2(spVct2);
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
	
	static public boolean isNumber(String token) {
		return token.matches("\\d+");
	}
	
	static public void randomize(double[] pros, double beta) {
        double total = 0;
        Random r = new Random();
        for (int i = 0; i < pros.length; i++) {
            pros[i] = beta + r.nextDouble();//to avoid zero probability
            total += pros[i];
        }

        //normalize
        for (int i = 0; i < pros.length; i++)
            pros[i] = pros[i] / total;
    }
	
	static public String formatArray(double [] array) {
		StringBuffer buffer = new StringBuffer(256);
		for(int i=0;i<array.length;i++)
			if (i==0)
				buffer.append(Double.toString(array[i]));
			else
				buffer.append("," + Double.toString(array[i]));
		return String.format("(%s)", buffer.toString());
	}
	
	static public _SparseFeature[] createSpVct(HashMap<Integer, Double> vct) {
		_SparseFeature[] spVct = new _SparseFeature[vct.size()];
		
		int i = 0;
		Iterator<Entry<Integer, Double>> it = vct.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry<Integer, Double> pairs = (Map.Entry<Integer, Double>)it.next();
			double TF = pairs.getValue();
			spVct[i] = new _SparseFeature(pairs.getKey(), TF);
			i++;
		}
		Arrays.sort(spVct);		
		return spVct;
	}
	
	public static String cleanHTML(String content) {
		if (content.indexOf("<!--")==-1 || content.indexOf("-->")==-1)
			return content;//clean text
		
		int start = 0, end = content.indexOf("<!--");
		StringBuffer buffer = new StringBuffer(content.length());
		while(end!=-1) {
			if (end>start)
				buffer.append(content.substring(start, end).trim());
			start = content.indexOf("-->", end) + 3;
			end = content.indexOf("<!--", start);
		}
		
		if (start<content.length())
			buffer.append(content.substring(start));
		
		return cleanVideoReview(buffer.toString());
	}
	
	public static String cleanVideoReview(String content) {
		if (!content.contains("// <![CDATA[") || !content.contains("Length::"))
			return content;
		
		int start = content.indexOf("// <![CDATA["), end = content.indexOf("Length::", start);
		end = content.indexOf("Mins", end) + 4;
		StringBuffer buffer = new StringBuffer(content.length());
		buffer.append(content.substring(0, start));
		buffer.append(content.substring(end));
		
		if (buffer.length()==0)
			return null;
		else
			return buffer.toString();
	}
		
	public static boolean endWithPunct(String stn) {
		char lastChar = stn.charAt(stn.length()-1);
		return !((lastChar>='a' && lastChar<='z') 
				|| (lastChar>='A' && lastChar<='Z') 
				|| (lastChar>='0' && lastChar<='9'));
	}
}
