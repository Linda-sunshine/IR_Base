package structures;

import java.util.ArrayList;

/***
 * @author lin
 * The data structure stores the information of a user used in CoLinAdapt.
 */

public class _User {
	
	String m_userID;
	double[] m_lowDimRep;
	ArrayList<_Review> m_reviews; //The reviews the user have, they should be by ordered by time stamps.
	_SparseFeature[] m_x_sparse; //The BoW representation of a user.	
	ArrayList<_User> m_neighbors; //the neighbors of the current user.
	int m_reviewCount; //Record how many reviews have been used to update.
	
	public _User(String userID, ArrayList<_Review> reviews){
		m_userID = userID;
		m_reviews = reviews;	
		m_reviewCount = 0;
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
 }
