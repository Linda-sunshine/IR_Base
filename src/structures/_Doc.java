/**
 * 
 */
package structures;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
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
	private int m_totalLength; //The total length of the document.
	
	private int m_y_label; // classification target, that is the index of the labels.
	private int m_predict_label; //The predicted result.
	private double m_y_value; // regression target, like linear regression only has one value.	
	private long m_timeStamp; //The timeStamp for this review.
	
	//We only need one representation between dense vector and sparse vector: V-dimensional vector.
	private _SparseFeature[] m_x_sparse; // sparse representation of features: default value will be zero.
	
	//Constructor.
	public _Doc(){
		this.m_predict_label = 0;
		this.m_totalLength = 0;
		//this.m_timeStamp = 0;
	}
	
	//Constructor.
	public _Doc (int ID, String source, int ylabel){
		this.m_ID = ID;
		this.source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
	}
	
	public _Doc (int ID, String source, int ylabel, long timeStamp){
		this.m_ID = ID;
		this.source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		this.m_timeStamp = timeStamp;
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
	
	//Get the time stamp of the document.
	public long getTimeStamp(){
		return this.m_timeStamp;
	}
	
	//Set the time stamp for the document.
	public void setTimeStamp(long t){
		this.m_timeStamp = t;
	}
	
	//Get the sparse vector of the document.
	public _SparseFeature[] getSparse(){
		return this.m_x_sparse;
	}
	
	//return the unique number of features in the doc
	public int getDocLength() {
		return this.m_x_sparse.length;
	}	
	
	//Get the total number of tokens in a document.
	public int getTotalDocLength(){
		return this.m_totalLength;
	}
		
	//Given an index, find the corresponding value of the feature. 
	public double findValueWithIndex(int index){
		for(_SparseFeature sf: this.m_x_sparse){
			if(index == sf.getIndex())
				return sf.getValue();
		}
		return 0;
	}
	
	//Create the sparse vector for the document.
	public _SparseFeature[] createSpVct(HashMap<Integer, Double> spVct) {
		int i = 0;
		m_x_sparse = new _SparseFeature[spVct.size()];
		Iterator<Entry<Integer, Double>> it = spVct.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry<Integer, Double> pairs = (Map.Entry<Integer, Double>)it.next();
			double TF = pairs.getValue();
			this.m_x_sparse[i] = new _SparseFeature(pairs.getKey(), TF);
			m_totalLength += TF;
			i++;
		}
		Arrays.sort(m_x_sparse);
		return m_x_sparse;
	}
	
	//Create a sparse vector with time features.
	public void createSpVctWithTime(LinkedList<_Doc> preDocs, int featureSize, double norm){
		int featureLength = this.m_x_sparse.length;
		int timeLength = preDocs.size();
		_SparseFeature[] tempSparse = new _SparseFeature[featureLength + timeLength];
		System.arraycopy(m_x_sparse, 0, tempSparse, 0, featureLength);		
		int count = 0;
		for(_Doc doc:preDocs){
			int index = featureSize + count;
			double value = norm * doc.getYLabel();
			//value *= Utils.calculateSimilarity(doc.getSparse(), m_x_sparse);
			tempSparse[featureLength + count] = new _SparseFeature(index, value);
			count++;
		}		
		this.m_x_sparse = tempSparse;
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
}
