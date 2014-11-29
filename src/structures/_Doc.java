/**
 * 
 */
package structures;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

/**
 * @author lingong
 * General structure to present a document for DM/ML/IR
 */
public class _Doc {
	
	private String m_name; // name of this document
	private int m_ID; // unique id of the document in the collection
	private String source; //The content of the source file.
	private int m_y_label; // classification target, that is the index of the labels.
	private int m_predict_label; //The predicted result.
	private double m_influence; //The influence value of the document.
	private double m_y_value; // regression target, like linear regression only has one value.
	private int m_totalLength; //The total length of the document.
	private int m_timeStamp;
	
	//We only need one representation between dense vector and sparse vector: V-dimensional vector.
	private _SparseFeature[] m_x_sparse; // sparse representation of features: default value will be zero.
	
	//Constructor.
	public _Doc(){
		this.m_predict_label = 0;
		this.m_influence = 0;
		this.m_totalLength = 0;
		this.m_timeStamp = 0;
	}
	
	//Constructor.
	public _Doc (int ID, String source, int ylabel){
		this.m_ID = ID;
		this.source = source;
		this.m_y_label = ylabel;
		this.m_influence = 0;
		this.m_totalLength = 0;
		this.m_timeStamp = 0;
	}
	
	//Get the name of the document.
	public String getName(){
		return this.m_name;
	}
	
	//Set a new name for the document.
	public String setName(String name){
		this.m_name = name;
		return this.m_name;
	}
	
	//Get the ID of the document.
	public int getID(){
		return this.m_ID;
	}
	
	//Set a new ID for the document.
	public int setID(int id){
		this.m_ID = id;
		return this.m_ID;
	}
	
	//Get the source content of a document.
	public String getSource(){
		return this.source;
	}
	
	//Get the real label of the doc.
	public int getYLabel() {
		return this.m_y_label;
	}
	
	//Set the Y value for the document, Y represents the class.
	public int setYLabel(int label){
		this.m_y_label = label;
		return this.m_y_label;
	}
	
	//Get the Y value, such as the result of linear regression.
	public double getYValue(){
		return this.m_y_value;
	}
	
	//Get the sparse vector of the document.
	public _SparseFeature[] getSparse(){
		return this.m_x_sparse;
	}
	
	//return the unique number of features in the doc
	public int getDocLength() {
		return this.m_x_sparse.length;
	}
	
	//Set the document's length, which is the total number of tokens.
	public void setTotalLength(int l){
		this.m_totalLength = l;
	}
	
	//Get the total number of tokens in a document.
	public int getTotalDocLength(){
		return this.m_totalLength;
	}
		
	//Given an index, find the corresponding value of the feature. 
	public double findValueWithIndex(int index){
		double value = 0;
		for(_SparseFeature sf: this.m_x_sparse){
			if(index == sf.getIndex()){
				value = sf.getValue();
			} else value = 0;
		}
		return value;
	}
	
	public void setInfluence(double influence){
		this.m_influence = influence;
	}
	
	//Create the sparse vector for the document.
	public void createSpVct(HashMap<Integer, Double> spVct) {
		int i = 0;
		m_x_sparse = new _SparseFeature[spVct.size()];
		Iterator<Entry<Integer, Double>> it = spVct.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry<Integer, Double> pairs = (Map.Entry<Integer, Double>)it.next();
			_SparseFeature sf = new _SparseFeature();
			sf.setIndex((int) pairs.getKey());
			sf.setValue((double) pairs.getValue());
			if(i < m_x_sparse.length){
				this.m_x_sparse[i] = sf;
				i++;
			} 
			else
				System.out.println("Error!! The index of sparse array out of bound!!");
		}
		Arrays.sort(m_x_sparse);
	}
	
	//Create the sparse vector for the document.
	public void createSpVctWithTime(HashMap<Integer, Double> spVct, int timeIndex) {
		int i = 0;
		m_x_sparse = new _SparseFeature[spVct.size() + 1];
		Iterator<Entry<Integer, Double>> it = spVct.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<Integer, Double> pairs = (Map.Entry<Integer, Double>) it.next();
			_SparseFeature sf = new _SparseFeature();
			sf.setIndex((int) pairs.getKey());
			sf.setValue((double) pairs.getValue());
			if (i < m_x_sparse.length) {
				this.m_x_sparse[i] = sf;
				i++;
			} else
				System.out.println("Error!! The index of sparse array out of bound!!");
		}
		_SparseFeature Influence = new _SparseFeature();
		Influence.setIndex(timeIndex);
		Influence.setValue(m_influence);
		m_x_sparse[spVct.size() + 1] = Influence;
		Arrays.sort(m_x_sparse);
		}	
	
	//Get the predicted result, which is used for comparison.
	public int getPredictLabel() {
		return this.m_predict_label;
	}
	
	//Set the predict result back to the doc.
	public int setPredictLabel(int label){
		this.m_predict_label = label;
		return this.m_predict_label;
	}
	
	//Calculate the sum of all the values of features.
	public double sumOfFeatures(_SparseFeature[] fs) {
		double sum = 0;
		for (_SparseFeature feature: fs){
			double value = feature.getValue();
			sum += value;
		}
		return sum;
	}
	
	//L2 = sqrt(sum of fsValue*fsValue).
	public double sumOfFeaturesL2(_SparseFeature[] fs) {
		double sum = 0;
		for (_SparseFeature feature: fs){
			double value = feature.getValue();
			sum += value * value;
		}
		return Math.sqrt(sum);
	}
	
	//L1 normalization.
	//Set the normalized value back to the sparse feature.
	public void L1Normalization(_SparseFeature[] fs) {
		double sum = sumOfFeatures(fs);
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
	
	//L2 normalization.
	public void L2Normalization(_SparseFeature[] fs) {
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
}
