package structures;

import java.util.ArrayList;
import java.util.TreeMap;

import CoLinAdapt.CoLinAdapt;
import CoLinAdapt.LinAdapt;

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
	protected ArrayList<Double> m_neighborSims;
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
	
	public void setCoefficients(double shift, double scale, double r2){
		m_coLinAdapt.setCoefficients(shift, scale, r2);
	}
	public int getIndex(){
		return m_userIndex;
	}
	//Return the linAdapt model.
	public LinAdapt getLinAdapt(){
		return m_linAdapt;
	}
	
	//Return the coLinAdapt model.
	public CoLinAdapt getCoLinAdapt(){
		return m_coLinAdapt;
	}
	// Get the user ID.
	public String getUserID(){
		return m_userID;
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
	
	public ArrayList<_Review> getReviews(){
		return m_reviews;
	}
	
	public int getReviewSize(){
		return m_reviews.size();
	}
	
	//Construct the neigbors for the current user.
	public void setNeighbors(ArrayList<_User> neighbors){
		m_neighbors = neighbors;
	}
	public void setNeighborSims(ArrayList<Double> sims){
		m_neighborSims = sims;
	}
	public void setNeighborIndexes(ArrayList<Integer> indexes){
		m_neighborIndexes = indexes;
	}
	
	//Given neighbors, pass them to model.
	public void setCoLinAdaptNeighbors(){
		m_coLinAdapt.setNeighbors(m_neighbors);
	}
	public void setCoLinAdaptNeighborIndexes(){
		m_coLinAdapt.setNeighborIndexes(m_neighborIndexes);
	}
	public void setCoLinAdpatNeighborSims(){
		m_coLinAdapt.setNeighborSims(m_neighborSims);
	}

	public ArrayList<_User> getNeighbors(){
		return m_neighbors;
	}
	
	public int[] getNeighborIndexes(){
		int[] indexes = new int[m_neighbors.size()];
		for(int i=0; i<m_neighbors.size(); i++)
			indexes[i] = m_neighbors.get(i).getIndex();
		return indexes;
	}
	
	public ArrayList<Double> getNeighborSims(){
		return m_neighborSims;
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
