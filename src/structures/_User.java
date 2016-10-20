package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Review.rType;
import utils.Utils;

/***
 * @author lin
 * The data structure stores the information of a user used in CoLinAdapt.
 */

public class _User {
	
	protected String m_userID;
	
	//text reviews associated with this user
	protected ArrayList<_Review> m_reviews; //The reviews the user have, they should be by ordered by time stamps.
	
	//profile for this user
	protected double[] m_lowDimProfile;
	protected _SparseFeature[] m_BoWProfile; //The BoW representation of a user.
	
	//personalized prediction model
	protected double[] m_pWeight;
	protected int m_classNo;
	protected int m_featureSize;
	protected int[] m_category;
	protected double[] m_svmWeights;
	
	protected double m_sim; // Similarity of itself.
	
	public void setSVMWeights(double[] weights){
		m_svmWeights = new double[weights.length];
		m_svmWeights = Arrays.copyOf(weights, weights.length);
	}
	
	public double[] getSVMWeights(){
		return m_svmWeights;
	}
	// performance statistics
	_PerformanceStat m_perfStat;
	
	int m_cIndex = 0; // added by Lin, cluster number.
	
	public _User(int cindex, int classNo){
		m_cIndex = cindex;
		m_classNo = classNo;
		m_reviews = new ArrayList<_Review>();		

		m_userID = null;
		m_lowDimProfile = null;
		m_BoWProfile = null;
		m_pWeight = null;
		
		m_perfStat = new _PerformanceStat(classNo);
	}
	
	public _User(String userID, int classNo, ArrayList<_Review> reviews){
		m_userID = userID;
		m_reviews = reviews;
		m_classNo = classNo;

		m_lowDimProfile = null;
		m_BoWProfile = null;
		m_pWeight = null;

		m_perfStat = new _PerformanceStat(classNo);
		
		constructSparseVector();
		calcPosRatio();
	}
	
	public _User(String userID, int classNo, ArrayList<_Review> reviews, int[] category){
		m_userID = userID;
		m_reviews = reviews;
		m_classNo = classNo;

		m_lowDimProfile = null;
		m_BoWProfile = null;
		m_pWeight = null;

		m_perfStat = new _PerformanceStat(classNo);
		m_category = category;
		constructSparseVector();
	}
	// added by Lin for setting the index of user cluster.
	public void setClusterIndex(int i) {
		m_cIndex = i;
	}
	// added by Lin for accessing the index of user cluster.
	public int getClusterIndex() {
		return m_cIndex;
	}
	// Get the user ID.
	public String getUserID(){
		return m_userID;
	}
	
	public String toString() {
		return String.format("%s-R:%d", m_userID, getReviewSize());
	}
	
	public boolean hasAdaptationData() {
		for(_Review r:m_reviews) {
			if (r.getType() == rType.ADAPTATION) {
				return true;
			}
		}
		return false;
	}
	
	public void initModel(int featureSize) {
		m_pWeight = new double[featureSize];
	}
	
	public void setModel(double[] weight) {
		initModel(weight.length);
		System.arraycopy(weight, 0, m_pWeight, 0, weight.length);
		m_featureSize = weight.length;
	}

	public double[] getPersonalizedModel() {
		return m_pWeight;
	}
	
	public void constructSparseVector(){
		ArrayList<_SparseFeature[]> reviews = new ArrayList<_SparseFeature[]>();

		for(_Review r: m_reviews) 
			reviews.add(r.getSparse());
		
		m_BoWProfile = Utils.MergeSpVcts(reviews);// this BoW representation is not normalized?!
	}
	
	//Get the sparse vector of the user.
	public _SparseFeature[] getBoWProfile(){
		return m_BoWProfile;
	}
	
	public int getReviewSize() {
		return m_reviews==null?0:m_reviews.size();
	}
	
	public ArrayList<_Review> getReviews(){
		return m_reviews;
	}
	
	public double getBoWSim(_User u) {
		return Utils.cosine(m_BoWProfile, u.getBoWProfile());
	}
	
	public double getBoWSimBaseSVMWeights(_User u){
		return Utils.cosine(m_svmWeights, u.getSVMWeights());
	}
	
	public double getSVDSim(_User u) {
		return Utils.cosine(u.m_lowDimProfile, m_lowDimProfile);
	}
	
	public double linearFunc(_SparseFeature[] fvs, int classid) {
		return Utils.dotProduct(m_pWeight, fvs, classid*m_featureSize);
	}
	
	public int predict(_Doc doc) {
		_SparseFeature[] fv = doc.getSparse();

		double maxScore = Utils.dotProduct(m_pWeight, fv, 0);
		if (m_classNo==2) {
			return maxScore>0?1:0;
		} else {//we will have k classes for multi-class classification
			double score;
			int pred = 0; 
		
			for(int i = 1; i < m_classNo; i++) {
				score = Utils.dotProduct(m_pWeight, fv, i * (m_featureSize + 1));
				if (score>maxScore) {
					maxScore = score;
					pred = i;
				}
			}
			return pred;
		}
	}
	
	public void addOnePredResult(int predL, int trueL){
		m_perfStat.addOnePredResult(predL, trueL);
	}
	
	public _PerformanceStat getPerfStat() {
		return m_perfStat;
	}
	
	// Added by Lin for lowDimProfile.
	public void setLowDimProfile(double[] ld){
		m_lowDimProfile = ld;
	}
	
	// Added by Lin to access the low dim profile.
	public double[] getLowDimProfile(){
		return m_lowDimProfile;
	}
	
	public double calculatePosRatio(){
		double count = 0;
		for(_Review r: m_reviews){
			if(r.getYLabel() == 1)
				count++;
		}
		return count/m_reviews.size();
	}
	
	public int[] getCategory(){
		return m_category;
	}

	double m_avgIDF = 0;
	// Set average IDF value.
	public void setAvgIDF(double v){
		m_avgIDF = v;
	}
	
	public void setSimilarity(double sim){
		m_sim = sim;
	}
	
	public void appendRvws(ArrayList<_Review> rs){
		for(_Review r: rs)
			m_reviews.add(r);
	}
	
	double m_posRatio = 0;
	public void calcPosRatio(){
		double pos = 0;
		for(_Review r: m_reviews){
			if(r.getYLabel() == 1)
				pos++;
		}
		m_posRatio = pos / m_reviews.size();
	}
	
	public double getPosRatio(){
		return m_posRatio;
	}
	
	// Added by Lin for accumulating super user.
	public void mergeReviews(ArrayList<_Review> reviews) {
		m_reviews.addAll(reviews);
	}

	// Added by Lin for kmeans based on profile.
	public int[] getProfIndices() {
		int[] indices = new int[m_BoWProfile.length];
		for (int i = 0; i < m_BoWProfile.length; i++)
			indices[i] = m_BoWProfile[i].m_index;

		return indices;
	}
	public double[] getProfValues() {
		double[] values = new double[m_BoWProfile.length];
		for(int i=0; i<m_BoWProfile.length; i++) 
			values[i] = m_BoWProfile[i].m_value;
		
		return values;
	}
	
	// added by Lin for CF.
	
	/**added by Lin for cf.**/
	private HashMap<String, Integer> m_itemIDRating = new HashMap<String, Integer>(); //This hashmap contains all the items the user purchased and corresponding ratings.
	private double m_nDCG;
	private double m_MAP;
	
	/***added by Lin for cf***/
	public void setNDCG(double d){
		m_nDCG = d;
	}
	public void setMAP(double m){
		m_MAP = m;
	}
	public double getNDCG(){
		return m_nDCG;
	}
	public double getMAP(){
		return m_MAP;
	}
	public void addOneItemIDRatingPair(String item, int r){
		m_itemIDRating.put(item, r);
	}
	public HashMap<String, Integer> getItemIDRating(){
		return m_itemIDRating;
	}
	
	public void removeOneReview(String prodID){
		int index = 0;
		for(_Review r: m_reviews){
			if(r.getItemID().equals(prodID)){
				index = m_reviews.indexOf(r);
				break;
			}
		}
		m_reviews.remove(index);
	}
 }