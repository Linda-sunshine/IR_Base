/**
 * 
 */
package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;

import structures._Doc;
import structures._PerformanceStat;
import structures._Review;
import structures._User;
import structures._Review.rType;

/**
 * @author Hongning Wang
 * shared adaptation structure
 */
public class _AdaptStruct {

	protected int m_id = 0; // by default all users have the same user ID
	protected _User m_user; // unit to store train/adaptation/test data and final personalized model
	
	//structures for online model update
	protected LinkedList<_Review> m_adaptCache;// adaptation cache to hold most recent observations, default size is one
	protected int m_cacheSize = 3;
	protected int m_adaptPtr, m_adaptStartPos, m_adaptEndPos;		
	protected double m_updateCount;
		
	public _AdaptStruct(_User user) {
		m_user = user;
		
		//trace to the first and last adaptation instance
		//we assume the instances are ordered and then separate into [train, adapt, test]
		m_adaptStartPos = -1;
		m_adaptEndPos = -1;
		ArrayList<_Review> reviews = user.getReviews();
		for(int i=0; i<reviews.size(); i++) {
			if (reviews.get(i).getType() == rType.ADAPTATION) {
				if (m_adaptStartPos==-1)
					m_adaptStartPos = i;
				m_adaptEndPos = i;
			}
		}
		
		if (m_adaptEndPos!=-1)
			m_adaptEndPos ++;//point to the next testing data or the end of reviews
		
		resetAdaptPtr();
	}
	
	@Override
	public String toString() {		
		return String.format("%d-A:%d-T:%d", m_id, getAdaptationSize(), getTestSize());
	}
	
	public String getUserID() {
		return m_user.getUserID();
	}

	public void resetAdaptPtr() {
		m_updateCount = 0;
		m_adaptPtr = m_adaptStartPos;
		if (m_adaptCache==null) 
			m_adaptCache = new LinkedList<_Review>();
		else
			m_adaptCache.clear();				
	}
	
	//return all the reviews
	public ArrayList<_Review> getReviews(){
		return m_user.getReviews();
	}
	
	//total number of adaptation reviews
	public int getAdaptationSize() {
		return m_adaptEndPos - m_adaptStartPos;
	}
	
	public void incUpdatedCount(double inc) {
		m_updateCount += inc;
	}
	
	public double getUpdateCount() {
		return m_updateCount;
	}
	
	public int getTestSize() {
		return m_user.getReviewSize() - m_adaptEndPos;
	}
	
	//already utilized adaptation reviews
	public int getAdaptedCount() {
		return m_adaptPtr - m_adaptStartPos;
	}
	
	// Get a mini-batch of reviews for online learning
	public Collection<_Review> nextAdaptationIns(){
		ArrayList<_Review> reviews = m_user.getReviews();
		
		//reach the maximum storage
		if (m_adaptCache.size()>=m_cacheSize)
			m_adaptCache.poll();
		
		if(m_adaptPtr < m_adaptEndPos){
			m_adaptCache.add(reviews.get(m_adaptPtr));
			m_adaptPtr++; //Move to the next review of the current user.
		}
		return m_adaptCache;
	}
	
	// Get a mini-batch of reviews for online learning without moving the adaptation pointer forward
	public Collection<_Review> getAdaptationCache(){
		return m_adaptCache;
	}
	
	public int getAdaptationCacheSize() {
		return m_adaptCache.size();
	}	
	
	public _Review getLatestTestIns() {
		ArrayList<_Review> reviews = m_user.getReviews();
		if(m_adaptPtr < m_adaptEndPos)
			return reviews.get(m_adaptPtr);
		else
			return null;
	}
	
	public boolean hasNextAdaptationIns() {
		return m_adaptPtr < m_adaptEndPos;
	}
	
	public void setPersonalizedModel(double[] pWeight, int classNo, int featureSize) {
		m_user.setModel(pWeight, classNo, featureSize);
	}
	
	public int predict(_Doc doc) {
		return m_user.predict(doc);
	}
	
	public _PerformanceStat getPerfStat() {
		return m_user.getPerfStat();
	}
	
	public double[] getPWeights() {
		return m_user.getPersonalizedModel();
	}
}
