package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;
import structures._User;

public class CategoryAnalyzer extends MultiThreadedUserAnalyzer {

	protected boolean m_ctgFlag = false; // Whether category is loaded or not.
	protected int[] m_ctgCounts;
	protected int m_ctgThreshold = Integer.MAX_VALUE; // added by Lin, the category threshold for selecting users.
	HashMap<Integer, ArrayList<_Review>> m_ctgRvws = new HashMap<Integer, ArrayList<_Review>>();
	protected ArrayList<String> m_categories;
	protected int m_start = 0, m_end = Integer.MAX_VALUE; // Added by Lin for filtering reviews.
	protected double m_globalLen = 0, m_maxLen = 0;
	
	public CategoryAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold, int numberOfCores)
					throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores);
		initCtgRvw();
	}
	
	public void initCtgRvw(){
		m_ctgRvws.put(5, new ArrayList<_Review>());
		m_ctgRvws.put(19, new ArrayList<_Review>());
		m_ctgRvws.put(24, new ArrayList<_Review>());
		m_ctgRvws.put(25, new ArrayList<_Review>());
		m_ctgRvws.put(38, new ArrayList<_Review>());
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
	public void loadUser(String filename, int core){
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
			int ylabel, cIndex = 0;
			long timestamp;
			double localSize = 0, localLength = 0, localAvg = 0;
			while((line = reader.readLine()) != null){
				productID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				if(m_categories.contains(category)){
					cIndex = m_categories.indexOf(category);
					m_ctgCounts[cIndex]++;
				}
				
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
						if(cIndex == 5 || cIndex == 19 || cIndex == 24 || cIndex == 25 || cIndex == 38)
							m_ctgRvws.get(cIndex).add(review);
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
	
	public HashMap<Integer, ArrayList<_Review>> getCtgRvws(){
		return m_ctgRvws;
	}
//	// Sample a specific number of reviews from the given 2500 reviews.
//	public ArrayList<_Doc> sample(ArrayList<_Doc> rs, int no){
//		Set<Integer> indexes = new HashSet<Integer>();
//		while(indexes.size() < no){
//			indexes.add((int) (rs.size()*Math.random()));
//		}
//		ArrayList<_Doc> samples = new ArrayList<_Doc>();
//		for(int i: indexes){
//			samples.add(rs.get(i));
//		}
//		return samples;
//	}
	
	public ArrayList<_Review> sample(ArrayList<_Review> rs, int no){
		Set<Integer> indexes = new HashSet<Integer>();
		while(indexes.size() < no){
			indexes.add((int) (rs.size()*Math.random()));
		}
		ArrayList<_Review> samples = new ArrayList<_Review>();
		for(int i: indexes){
			samples.add(rs.get(i));
		}
		return samples;
	}
	// Split the samples to train set and test set.
	ArrayList<_Doc> m_trainSet, m_testSet;
	public void split(ArrayList<_Review> rs, int no){
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		int[] flags = new int[rs.size()];
		Set<Integer> indexes = new HashSet<Integer>();
		while(indexes.size() < no){
			indexes.add((int) (rs.size()*Math.random()));
		}
		for(int index: indexes)	
			flags[index] = 1;// test set.
		
		for(int i=0; i<flags.length; i++){
			if(flags[i] == 0)
				m_trainSet.add(rs.get(i));
			else 
				m_testSet.add(rs.get(i));
		}
	}
	
	public ArrayList<_Doc> getTrainSet(){
		return m_trainSet;
	}
	
	public ArrayList<_Doc> getTestSet(){
		return m_testSet;
	}

}
