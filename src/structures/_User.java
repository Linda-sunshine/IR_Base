package structures;

import java.util.ArrayList;

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
	protected double[] m_pWeightB;
	protected int m_classNo;
	protected int m_featureSize;
	
	// performance statistics
	_PerformanceStat m_perfStat;
	
	public _User(String userID, int classNo, ArrayList<_Review> reviews){
		m_userID = userID;
		m_reviews = reviews;
		m_classNo = classNo;

		m_lowDimProfile = null;
		m_BoWProfile = null;
		m_pWeight = null;

		m_perfStat = new _PerformanceStat(classNo);
		
		constructSparseVector();
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
	// added by Lin.
	public void setModelB(double[] weight){
		m_pWeightB = new double[m_featureSize];
		System.arraycopy(weight, 0, m_pWeightB, 0, weight.length);
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
	
	public double getSVDSim(_User u) {
		return Utils.cosine(u.m_lowDimProfile, m_lowDimProfile);
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
 }