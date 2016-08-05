package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._User;



public class CFMultiThreadedUserAnalyzer extends MultiThreadedUserAnalyzer {
	public CFMultiThreadedUserAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold, int numberOfCores)
					throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores);
	}
	
	// Load one file as a user here. 
	public void loadOneUser(String filename, int core){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.
			// Skip the first line since it is user name.
			reader.readLine(); 

			String productID, source, category;
			ArrayList<_Review> reviews = new ArrayList<_Review>();
			_Review review;
			int ylabel;
			long timestamp;
			double localSize = 0, localLength = 0, localAvg = 0;
			while((line = reader.readLine()) != null){
				productID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category				
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());

				// Construct the new review.
				ylabel--;
				review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, productID, category, timestamp);
				if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
					reviews.add(review);
					localLength += review.getDocLength();
					localSize++;
				}
			}
			localAvg = localLength / localSize;
			
			// Added by Lin for debugging.
			if(reviews.size() > 1 && (localAvg < m_end) && (localAvg > m_start)){//at least one for adaptation and one for testing
				if( localAvg > m_maxLen)
					m_maxLen = localLength / localSize;
				m_globalLen += localLength;
				synchronized (m_allocReviewLock) {
					allocateReviews(reviews);			
					m_users.add(new _User(userID, m_classNo, reviews));
				}
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
