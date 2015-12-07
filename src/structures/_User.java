package structures;

import java.util.ArrayList;

import CoLinAdapt.CoLinAdapt;
import CoLinAdapt.LinAdapt;
import utils.Utils;

/***
 * @author lin
 * The data structure stores the information of a user used in CoLinAdapt.
 */

public class _User {
	
	protected String m_userID;
	protected int m_userIndex;
	
	//text reviews associated with this user
	protected ArrayList<_Review> m_reviews; //The reviews the user have, they should be by ordered by time stamps.
	
	//profile for this user
	protected double[] m_lowDimRep;
	protected _SparseFeature[] m_x_sparse; //The BoW representation of a user.
	
	//neighborhood for this user
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
	}
	
	public void initCoLinAdapt(int fg, int fn, double[] globalWeights, int[] featureGroupIndexes){
		m_coLinAdapt = new CoLinAdapt(fg, fn, globalWeights, featureGroupIndexes);
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
		ArrayList<_SparseFeature[]> reviews = new ArrayList<_SparseFeature[]>();

		for(_Review r: m_reviews) 
			reviews.add(r.getSparse());
		
		m_x_sparse = Utils.MergeSpVcts(reviews);
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
	
	//Construct the neighbors for the current user.[need to be implemented]
//	public void constructNeighbors(){
//		
//	}
	
	public void transferNeighbors2Model(){
		m_coLinAdapt.setNeighbors(m_neighbors);
	}
	
	public int getIndex(){
		return m_userIndex;
	}
	
	//Construct the neighbors for the current user.
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
	
	public int[] getNeighborIndexes(){
		int[] indexes = new int[m_neighbors.size()];
		for(int i=0; i<m_neighbors.size(); i++)
			indexes[i] = m_neighbors.get(i).getIndex();
		return indexes;
	}
	
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
