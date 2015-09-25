/**
 * 
 */
package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Random;

import utils.Utils;

/**
 * @author lingong
 * General structure to present a document for DM/ML/IR
 */
public class _Doc implements Comparable<_Doc> {
	String m_name;
	int m_ID; // unique id of the document in the collection
	String m_itemID; // ID of the product being commented
	String m_title; //The short title of the review.
	
	String m_source; //The content of the source file.
	int m_sourceType = 1; // source is 1 for Amazon and 2 for newEgg
	int m_totalLength; //The total length of the document in tokens
	
	int m_y_label; // classification target, that is the index of the labels.
	int m_predict_label; //The predicted result.
	long m_timeStamp; //The timeStamp for this review.
	
	// added by Lin
	private _SparseFeature[] m_x_posVct; // sparse vector projected by pos tagging 
	private double[] m_x_aspVct; // aspect vector	
		
	//only used in learning to rank for random walk
	double m_weight = 1.0; // instance weight for supervised model training (will be reset by PageRank)
	double m_stopwordProportion = 0;
	double m_avgIDF = 0;
	double m_sentiScore = 0; //Sentiment score from sentiwordnet
	public boolean forTest = false; // true mean it is for test
	
	public void setSourceType(int sourceName) {
		m_sourceType = sourceName;
	}
	
	public int getSourceType() {
		return m_sourceType;
	}
	
	public double getAvgIDF() {
		return m_avgIDF;
	}

	public void setAvgIDF(double avgIDF) {
		this.m_avgIDF = avgIDF;
	}

	public double getStopwordProportion() {
		return m_stopwordProportion;
	}

	public void setStopwordProportion(double stopwordProportion) {
		this.m_stopwordProportion = stopwordProportion;
	}

	public void setSentiScore(double s){
		this.m_sentiScore = s;
	}
	
	public double getSentiScore(){
		return this.m_sentiScore;
	}

	//We only need one representation between dense vector and sparse vector: V-dimensional vector.
	private _SparseFeature[] m_x_sparse; // sparse representation of features: default value will be zero.
	private _SparseFeature[] m_x_projection; // selected features for similarity computation (NOTE: will use different indexing system!!)	
	
	static public final int stn_fv_size = 4; // bias, cosine, length_ratio, position, conjunction
	static public final int stn_senti_fv_size = 6; // bias, cosine, sentiWordNetScore, prior_positive_negative_count, POS tag divergency, conjunction
	
	_Stn[] m_sentences;
	
	//p(z|d) for topic models in general
	public double[] m_topics;
	//sufficient statistics for estimating p(z|d)
	public double[] m_sstat;//i.e., \gamma in variational inference p(\theta|\gamma) also used in EM Naive Bayes
	
	// structure only used by Gibbs sampling to speed up the sampling process
	public int[] m_words; 
	public int[] m_topicAssignment;
	
	// structure only used by variational inference
	public double[][] m_phi; // p(z|w, \phi)	
	Random m_rand;
	
	//Constructor.
	public _Doc (int ID, String source, int ylabel){
		this.m_ID = ID;
		this.m_source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		m_topics = null;
		m_sstat = null;
		m_words = null;
		m_topicAssignment = null;
		m_sentences = null;
	}
	
	public _Doc (int ID, String name, String prodID, String title, String source, int ylabel, long timeStamp){
		this.m_ID = ID;
		this.m_name = name;
		this.m_itemID = prodID;
		this.m_title = title;
		this.m_source = source;
		this.m_y_label = ylabel;
		this.m_totalLength = 0;
		this.m_timeStamp = timeStamp;
		m_topics = null;
		m_sstat = null;
		m_words = null;
		m_topicAssignment = null;
		m_sentences = null;
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
	public void setID(int id){
		this.m_ID = id;
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
	
	public String getTitle(){
		return m_title;
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
	public void setYLabel(int label){
		this.m_y_label = label;
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
	
	public _SparseFeature[] getProjectedFv() {
		return this.m_x_projection;
	}
	
	public int[] getIndices() {
		int[] indices = new int[m_x_sparse.length];
		for(int i=0; i<m_x_sparse.length; i++) 
			indices[i] = m_x_sparse[i].m_index;
		
		return indices;
	}
	
	public double[] getValues() {
		double[] values = new double[m_x_sparse.length];
		for(int i=0; i<m_x_sparse.length; i++) 
			values[i] = m_x_sparse[i].m_value;
		
		return values;
	}
	
	//return the unique number of features in the doc
	public int getDocLength() {
		return this.m_x_sparse.length;
	}	
	
	//Get the total number of tokens in a document.
	public int getTotalDocLength(){
		return this.m_totalLength;
	}
	
	void calcTotalLength() {
		m_totalLength = 0;
		for(_SparseFeature fv:m_x_sparse)
			m_totalLength += fv.getValue();
	}
	
	//Create the sparse vector for the document, taking value from different sections
	public void createSpVct(ArrayList<HashMap<Integer, Double>> spVcts) {
		m_x_sparse = Utils.createSpVct(spVcts);
		calcTotalLength();
	}
	
	//Create the sparse vector for the document.
	public void createSpVct(HashMap<Integer, Double> spVct) {
		m_x_sparse = Utils.createSpVct(spVct);
		calcTotalLength();
	}
	
	public void setSpVct(_SparseFeature[] x) {
		m_x_sparse = x;
		calcTotalLength();
	}
		
	//Create the sparse postagging vector for the document. 
	public void createPOSVct(HashMap<Integer, Double> posVct){
		m_x_posVct = Utils.createSpVct(posVct);
	}
	
	public _SparseFeature[] getPOSVct(){
		return m_x_posVct;
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
	public void setSentences(ArrayList<_Stn> stnList) {
		m_sentences = stnList.toArray(new _Stn[stnList.size()]);
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public int getSenetenceSize() {
		return this.m_sentences.length;
	}
	
	public _Stn[] getSentences() {
		return m_sentences;
	}
	
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	public _Stn getSentence(int index) {
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
	
	public boolean hasSegments() {
		return m_x_sparse[0].m_values != null;
	}
	
	//we will reset the topic vector
	public void setTopics(int k, double alpha) {
		if (m_topics==null || m_topics.length!=k) {
			m_topics = new double[k];
			m_sstat = new double[k];
		}
		Utils.randomize(m_sstat, alpha);
	}
	
	public double[] getTopics(){
		return m_topics;
	}
	
	//create necessary structure for variational inference
	public void setTopics4Variational(int k, double alpha) {
		if (m_topics==null || m_topics.length!=k) {
			m_topics = new double[k];
			m_sstat = new double[k];//used as p(z|w,\phi)
			m_phi = new double[m_x_sparse.length][k];
		}
		
		Arrays.fill(m_sstat, alpha);
		for(int n=0; n<m_x_sparse.length; n++) {
			Utils.randomize(m_phi[n], alpha);
			double v = m_x_sparse[n].getValue();
			for(int i=0; i<k; i++)
				m_sstat[i] += m_phi[n][i] * v;
		}
	}
	
	//create necessary structure to accelerate Gibbs sampling
	public void setTopics4Gibbs(int k, double alpha) {
		if (m_topics==null || m_topics.length!=k) {
			m_topics = new double[k];
			m_sstat = new double[k];
		}

		Arrays.fill(m_sstat, alpha);
		
		//Warning: in topic modeling, we cannot normalize the feature vector and we should only use TF as feature value!
		int docSize = (int)Utils.sumOfFeaturesL1(m_x_sparse);
		if (m_words==null || m_words.length != docSize) {
			m_topicAssignment = new int[docSize];
			m_words = new int[docSize];
		} 
		
		int wIndex = 0;
		if (m_rand==null)
			m_rand = new Random();
		for(_SparseFeature fv:m_x_sparse) {
			for(int j=0; j<fv.getValue(); j++) {
				m_words[wIndex] = fv.getIndex();
				m_topicAssignment[wIndex] = m_rand.nextInt(k); // randomly initializing the topics inside a document
				m_sstat[m_topicAssignment[wIndex]] ++; // collect the topic proportion
				
				wIndex ++;
			}
		}
	}
	
	//permutation the order of words for Gibbs sampling
	public void permutation() {
		int s, t;
		for(int i=m_words.length-1; i>1; i--) {
			s = m_rand.nextInt(i);
			
			//swap the word
			t = m_words[s];
			m_words[s] = m_words[i];
			m_words[i] = t;
			
			//swap the topic assignment
			t = m_topicAssignment[s];
			m_topicAssignment[s] = m_topicAssignment[i];
			m_topicAssignment[i] = t;
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
	
	public void setProjectedFv(Map<Integer, Integer> filter) {
		m_x_projection = Utils.projectSpVct(m_x_sparse, filter);
//		if (m_x_projection!=null)
//			Utils.L2Normalization(m_x_projection);
	}
	
	public void setProjectedFv(double[] denseFv) {
		m_x_projection = Utils.createSpVct(denseFv);
	}

	public void setAspVct(double[] aspVct){
		m_x_aspVct = aspVct;
	}
	
	public double[] getAspVct(){
		return m_x_aspVct;
	}

}
