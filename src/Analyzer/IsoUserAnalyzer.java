package Analyzer;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._Review.rType;

public class IsoUserAnalyzer extends MultiThreadedUserAnalyzer {

	int m_count = 0;// controls how many users will be used for training.
	Object m_countLock = new Object();
	int m_testThreshold = 1000;
	public IsoUserAnalyzer(String tokenModel, int classNo, String providedCV,
			int Ngram, int threshold, int numberOfCores, boolean b)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
	}
	public void setUserTestThreshold(int t){
		m_testThreshold = t;
	}
	
	@Override
	void allocateReviews(ArrayList<_Review> reviews) {
		Collections.sort(reviews);// sort the reviews by timestamp
		// Collect testing users.
		if(reviews.size() >= 4 && m_count < m_testThreshold){
			for(int i=0; i<reviews.size(); i++) {
				reviews.get(i).setType(rType.TEST);
				m_testSize ++;
			}
			synchronized(m_countLock){
				m_count++;
			}
		} else{
			// Collecting training users.
			int train = (int)(reviews.size() * m_trainRatio), adapt;
			if (m_enforceAdapt)
				adapt = Math.max(1, (int)(reviews.size() * (m_trainRatio + m_adaptRatio)));
			else
				adapt = (int)(reviews.size() * (m_trainRatio + m_adaptRatio));
		
			for(int i=0; i<reviews.size(); i++) {
				if (i<train) {
					reviews.get(i).setType(rType.TRAIN);
					m_trainSize ++;
				} else if (i<adapt) {
					reviews.get(i).setType(rType.ADAPTATION);
					m_adaptSize ++;
				} else {
					reviews.get(i).setType(rType.TEST);
					m_testSize ++;
				}
			}
		}
	}
}
