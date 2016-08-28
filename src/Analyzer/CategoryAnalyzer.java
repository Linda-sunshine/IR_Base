package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._User;

public class CategoryAnalyzer extends MultiThreadedUserAnalyzer {

	protected boolean m_ctgFlag = false; // Whether category is loaded or not.
	protected int[] m_ctgCounts;
	protected int m_ctgThreshold = Integer.MAX_VALUE; // added by Lin, the category threshold for selecting users.
	HashMap<String, ArrayList<_Review>> m_ctgRvws = new HashMap<String, ArrayList<_Review>>();

	public CategoryAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold, int numberOfCores)
					throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores);
	}
	
	// Added by Lin.
	public void loadCategory(String filename){
		m_categories = new ArrayList<String>();
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				m_categories.add(line);
			}
			reader.close();
			System.out.println(m_categories.size() + " categories are loaded.");
			m_ctgCounts = new int[m_categories.size()];
			m_ctgFlag = true;
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	// Added by Lin, set the threshold for category counts.
	public void setCtgThreshold(int k){
		m_ctgThreshold = k;
	}
	
	public void printCategory(){
		for(int i=0; i<m_categories.size(); i++){
			System.out.print(m_categories.get(i)+":"+m_ctgCounts[i]+"\t");
		}
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
//			if(m_ctgFlag)
//				m_ctgCounts = new int[m_categories.size()];
			ArrayList<_Review> reviews = new ArrayList<_Review>();
			_Review review;
			int ylabel;
			long timestamp;
			double localSize = 0, localLength = 0, localAvg = 0;
			while((line = reader.readLine()) != null){
				productID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				if(m_categories.contains(category))
					m_ctgCounts[m_categories.indexOf(category)]++;
				
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());

				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, productID, category, timestamp);
					if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
						reviews.add(review);
						localLength += review.getDocLength();
						localSize++;
					}
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
					m_users.add(new _User(userID, m_classNo, reviews));//, m_ctgCounts));
				}
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	


}
