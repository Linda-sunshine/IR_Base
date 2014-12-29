/**
 * 
 */
package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;

import utils.Utils;

/**
 * @author lingong
 * General structure to present a document for DM/ML/IR
 */
public class _Doc implements Comparable<_Doc> {
	String m_name;
	int m_ID; // unique id of the document in the collection
	String m_itemID; // ID of the product being commented
	
	String m_source; //The content of the source file.
	int m_totalLength; //The total length of the document.
	
	int m_y_label; // classification target, that is the index of the labels.
	int m_predict_label; //The predicted result.
	double m_y_value; // regression target, like linear regression only has one value.	
	long m_timeStamp; //The timeStamp for this review.
	
	double m_weight = 1.0; // instance weight for supervised model training (will be reset by PageRank)
	
	//We only need one representation between dense vector and sparse vector: V-dimensional vector.
	private _SparseFeature[] m_x_sparse; // sparse representation of features: default value will be zero.
	private _SparseFeature m_sentences [][]; // sentence array each row contains the unique word id 
	
	//p(z|d) for topic models in general
	public double[] m_topics;
	//sufficient statistics for estimating p(z|d)
	public double[] m_sstat;
	
	//Constructor.
	public _Doc (int ID, String source, int ylabel){
		this.m_ID = ID;
		this.m_source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		m_topics = null;
		m_sstat = null;
	}
	
	public _Doc (int ID, String source, int ylabel, long timeStamp){
		this.m_ID = ID;
		this.m_source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		this.m_timeStamp = timeStamp;
		m_topics = null;
		m_sstat = null;
	}
	
	public _Doc (int ID, String name, String source, String productID, int ylabel, long timeStamp){
		this.m_ID = ID;
		this.m_name = name;
		this.m_source = source;
		this.m_itemID = productID;
		
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		this.m_timeStamp = timeStamp;
		m_topics = null;
		m_sstat = null;
	}
	
	public void setWeight(double w) {
		m_weight = w;
	}
	
	public double getWeight() {
		return m_weight;
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
	
	public String getItemID() {
		return m_itemID;
	}
	
	public String getName() {
		return m_name;
	}
	
	//Get the source content of a document.
	public String getSource(){
		return this.m_source;
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
	
	//Create the sparse vector for the document.
	public void createSpVct(HashMap<Integer, Double> spVct) {
		m_x_sparse = Utils.createSpVct(spVct);
		for(_SparseFeature fv:m_x_sparse)
			m_totalLength += fv.getValue();
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
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public void setSentences(ArrayList<_SparseFeature[]> stnList) {
		m_sentences = stnList.toArray(new _SparseFeature [stnList.size()][]);
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public int getTotalSenetences() {
		return this.m_sentences.length;
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public _SparseFeature[] getSentences(int index) {
		return this.m_sentences[index];
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
	
	public void setTopics(int k, double beta) {
		if (m_topics==null || m_topics.length!=k) {
			m_topics = new double[k];
			m_sstat = new double[k];
		}
		Utils.randomize(m_topics, beta);
		Arrays.fill(m_sstat, 0);
	}
	
	public void clearSource() {
		m_source = null;
	}

	@Override
	public int compareTo(_Doc d) {
		int prodCompare = m_itemID.compareTo(d.m_itemID);
		if (prodCompare==0) {
			if(m_timeStamp == d.getTimeStamp())
				return 0;
			return m_timeStamp < d.getTimeStamp() ? -1 : 1;
		} else
			return prodCompare;
	}
	
	public boolean sameProduct(_Doc d) {
		return m_itemID.equals(d.m_itemID);
	}
	
	@Override
	public String toString() {
		return String.format("ProdID: %s\tID: %s\t Rating: %d\n%s", m_itemID, m_name, m_y_label, m_source);
	}
}
