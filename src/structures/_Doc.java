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
	private _SparseFeature[][] m_sentences; // sentence array each row contains the unique word id 
	
	static public final int stn_fv_size = 4; // cosine, length_ratio, position
	public double[][] m_sentence_features; // feature vector 	
	public double[] m_sentence_labels; // estimated transition labels p(\psi=1)
	
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
		m_sentence_features = null;
	}
	
	public _Doc (int ID, String source, int ylabel, long timeStamp){
		this.m_ID = ID;
		this.m_source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		this.m_timeStamp = timeStamp;
		m_topics = null;
		m_sstat = null;
		m_sentence_features = null;
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
		m_sentence_features = null;
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
	
	public void setItemID(String itemID) {
		m_itemID = itemID;
	}
	
	public String getItemID() {
		return m_itemID;
	}
	
	public void setName(String name) {
		m_name = name;
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
	
	public void setSpVct(_SparseFeature[] x) {
		m_x_sparse = x;
		//unable to know total length when loading from vector file
		m_totalLength = x.length;//this is totally inaccurate
	}
	
	//Create a sparse vector with time features.
	public void createSpVctWithTime(LinkedList<_Doc> preDocs, int featureSize, double movingAvg, double norm){
		int featureLength = this.m_x_sparse.length;
		int timeLength = preDocs.size();
		_SparseFeature[] tempSparse = new _SparseFeature[featureLength + timeLength + 1];//to include the moving average
		System.arraycopy(m_x_sparse, 0, tempSparse, 0, featureLength);		
		int count = 0;
		for(_Doc doc:preDocs){
			double value = norm * doc.getYLabel();
			value *= Utils.calculateSimilarity(doc.getSparse(), m_x_sparse);//time-based features won't be considered since m_x_sparse does not contain time-based features yet
			tempSparse[featureLength + count] = new _SparseFeature(featureSize + count, value);
			count++;
		}	
		tempSparse[featureLength + count] = new _SparseFeature(featureSize + count, movingAvg*norm);
		this.m_x_sparse = tempSparse;
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public void setSentences(ArrayList<_SparseFeature[]> stnList) {
		m_sentences = stnList.toArray(new _SparseFeature [stnList.size()][]);
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public int getSenetenceSize() {
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
		Arrays.fill(m_sstat, beta);
	}	
	
	public void setSentenceFeatureVector() {
		m_sentence_features = new double[this.getSenetenceSize()-1][stn_fv_size];
		m_sentence_labels = new double[this.getSenetenceSize()-1];
		
		// start from 2nd sentence
		double cLength, pLength = Utils.sumOfFeaturesL1(this.getSentences(0));
		double pSim = Utils.cosine(m_sentences[0], m_sentences[1]), nSim;
		int stnSize = this.getSenetenceSize();
		for(int i=1; i<stnSize; i++){
			//cosine similarity			
			m_sentence_features[i-1][0] = pSim;			
			
			cLength = Utils.sumOfFeaturesL1(this.getSentences(i));
			//length_ratio
			m_sentence_features[i-1][1] = (pLength-cLength)/Math.max(cLength, pLength);
			pLength = cLength;
			
			//position
			m_sentence_features[i-1][2] = (double)i / stnSize;
			
			//similar to previous or next
			if (i<stnSize-1) {
				nSim = Utils.cosine(m_sentences[i], m_sentences[i+1]);
				if (nSim>pSim)
					m_sentence_features[i-1][3] = 1;
				else if (nSim<pSim)
					m_sentence_features[i-1][3] = -1;
				pSim = nSim;
			}
		}
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
		if (m_itemID == null || d.m_itemID == null)
			return false;
		return m_itemID.equals(d.m_itemID);
	}
	
	@Override
	public String toString() {
		return String.format("ProdID: %s\tID: %s\t Rating: %d\n%s", m_itemID, m_name, m_y_label, m_source);
	}
}
