package structures;

import java.util.ArrayList;
import java.util.TreeMap;

import utils.Utils;

/***
 * @author lin
 * The data structure stores the information of a user used in CoLinAdapt.
 */

public class _User {
	
	protected String m_userID;
	protected int m_userIndex;
	protected double[] m_lowDimRep;
	protected ArrayList<_Review> m_reviews; //The reviews the user have, they should be by ordered by time stamps.
	protected _SparseFeature[] m_x_sparse; //The BoW representation of a user.	
	protected ArrayList<_User> m_neighbors; //the neighbors of the current user.
	protected ArrayList<Integer> m_neighborIndexes; // The indexes of neighbors.
	protected int m_reviewCount; //Record how many reviews have been used to update.
	protected LinAdapt m_linAdapt;
	protected CoLinAdapt m_coLinAdapt;
	
	public _User(String userID, ArrayList<_Review> reviews, int userIndex){
		m_userID = userID;
		m_reviews = reviews;	
		m_reviewCount = 0;
		m_userIndex = userIndex;
		constructSparseVector();
	}
	
	public void initLinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes){
		m_linAdapt = new LinAdapt(fg, fn, globalWeights, featureGroupIndexes);
		m_linAdapt.initA();
		m_linAdapt.initGradients();
	}
	
	public void initCoLinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes){
		m_coLinAdapt = new CoLinAdapt(fg, fn, globalWeights, featureGroupIndexes);
		m_coLinAdapt.initA();
	}
	
	public int getIndex(){
		return m_userIndex;
	}	
	//text reviews associated with this user
	protected ArrayList<_Review> m_reviews; //The reviews the user have, they should be by ordered by time stamps.
	
	//profile for this user
	protected double[] m_lowDimProfile;
	protected _SparseFeature[] m_BoWProfile; //The BoW representation of a user.
	
	//personalized prediction model
	protected double[] m_pWeight;
	protected int m_classNo;
	protected int m_featureSize;
	
	// performance statistics
	_PerformanceStat m_perfStat;
	
	public _User(String userID, int classNo, ArrayList<_Review> reviews){
		m_userID = userID;
		m_reviews = reviews;

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
	
	public void setModel(double[] weight, int classNo, int featureSize) {
		m_pWeight = new double[weight.length];
		System.arraycopy(weight, 0, m_pWeight, 0, weight.length);
		m_classNo = classNo;
		m_featureSize = featureSize;
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
	
	public void constructSparseVector(){
		TreeMap<Integer, Double> fvIndexValueMap = new TreeMap<Integer, Double>();
		double value;
		int index, count = 0;
		for(_Review r: m_reviews){
			for(_SparseFeature fv: r.getSparse()){
				index = fv.getIndex();
				if(!fvIndexValueMap.containsKey(index))
					fvIndexValueMap.put(index, fv.getValue());
				else{
					value = fvIndexValueMap.get(fv.getIndex()) + fv.getValue();
					fvIndexValueMap.put(index, value);
				}
			}
		}
		m_x_sparse = new _SparseFeature[fvIndexValueMap.size()];
		for(int i: fvIndexValueMap.keySet()){
			m_x_sparse[count++] = new _SparseFeature(i, fvIndexValueMap.get(i));
		}
	}
	
	//Get the sparse vector of the user.
	public _SparseFeature[] getSparse(){
		return m_x_sparse;
	}
	// Get one review from a user's reviews.
	public _Review getOneReview(){
		_Review rev = null;
		if(m_reviewCount < m_reviews.size()){
			rev = m_reviews.get(m_reviewCount);
			m_reviewCount++; //Move to the next review of the current user.
		}
		return rev;
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
	
	//Construct the neigbors for the current user.
	public void setNeighbors(ArrayList<_User> neighbors){
		m_neighbors = neighbors;
	}

	//Given neighbors, pass them to model.
	public void setCoLinAdaptNeighbors(){
		m_coLinAdapt.setNeighbors(m_neighbors);
	}
	
	public void setCoLinAdaptNeighborIndexes(ArrayList<Integer> indexes){
		m_neighborIndexes = indexes;
	}
	
	public void setCoLinAdpatNeighborSims(ArrayList<Double> sims){
		m_coLinAdapt.setNeighborSims(sims);
	}

	public ArrayList<_User> getNeighbors(){
		return m_neighbors;
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
	
	public int[] getNeighborIndexes(){
		int[] indexes = new int[m_neighbors.size()];
		for(int i=0; i<m_neighbors.size(); i++)
			indexes[i] = m_neighbors.get(i).getIndex();
		return indexes;
	}
	
//	public void initGradients4CoLinAdapt(){
//		m_coLinAdapt.initGradients();
//	}
	//Return the transformed matrix in LinAdapt.
	public double[] getLinAdaptA(){
		return m_linAdapt.getA();
	}
	
	public double[] getCoLinAdaptA(){
		return m_coLinAdapt.getA();
	}
	
	public void updateA(double[] newA){
		m_coLinAdapt.updateA(newA);
	}
 }
