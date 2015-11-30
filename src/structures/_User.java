package structures;

import java.util.ArrayList;

import CoLinAdapt.CoLinAdapt;
import CoLinAdapt.LinAdapt;

/***
 * @author lin
 * The data structure stores the information of a user used in CoLinAdapt.
 */

public class _User {
	
	protected String m_userID;
	protected double[] m_lowDimRep;
	protected ArrayList<_Review> m_reviews; //The reviews the user have, they should be by ordered by time stamps.
	protected _SparseFeature[] m_x_sparse; //The BoW representation of a user.	
	protected ArrayList<_User> m_neighbors; //the neighbors of the current user.
	protected int m_reviewCount; //Record how many reviews have been used to update.
	protected LinAdapt m_linAdapt;
	protected CoLinAdapt m_coLinAdapt;
	
	public _User(String userID, ArrayList<_Review> reviews){
		m_userID = userID;
		m_reviews = reviews;	
		m_reviewCount = 0;
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
	public void constructNeighbors(){
		
	}
	
	public void transferNeighbors2Model(){
		m_coLinAdapt.setNeighbors(m_neighbors);
	}
	public ArrayList<_User> getNeighbors(){
		return m_neighbors;
	}
 }
