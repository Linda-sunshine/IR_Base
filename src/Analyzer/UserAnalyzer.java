package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._Review.rType;
import structures._User;
import structures._stat;
import utils.Utils;

public class UserAnalyzer extends DocAnalyzer {
	
	ArrayList<_User> m_users; // Store all users with their reviews.
	double m_trainRatio = 0.25; // by default, the first 25% for training global model 
	double m_adaptRatio = 0.5; // by default, the next 50% for adaptation, and rest 25% for testing
	int m_trainSize = 0, m_adaptSize = 0, m_testSize = 0;
	
	public UserAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_users = new ArrayList<_User>();
	}
	
	public void config(double train, double adapt) {
		if (train<0 || train>1) {
			System.err.format("[Error]Incorrect setup of training ratio %.3f, which has to be in [0,1]\n", train);
			return;
		} else if (adapt<0 || adapt>1) {
			System.err.format("[Error]Incorrect setup of adaptation ratio %.3f, which has to be in [0,1]\n", adapt);
			return;
		} else if (train+adapt>1) {
			System.err.format("[Error]Incorrect setup of training and adaptation ratio (%.3f, %.3f), whose sum has to be in (0,1]\n", train, adapt);
			return;
		}
		
		m_trainRatio = train;
		m_adaptRatio = adapt;
	}
	
	//Load the features from a file and store them in the m_featurNames.@added by Lin.
	protected boolean LoadCV(String filename) {
		if (filename==null || filename.isEmpty())
			return false;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line, stats[];
			int ngram = 0, DFs[]= {0, 0};
			m_Ngram = 1;//default value of Ngram
			while ((line = reader.readLine()) != null) {
				stats = line.split(",");
				
				if (stats[1].equals("TOTALDF")) {
					m_TotalDF = (int)(Double.parseDouble(stats[2]));
				} else {
					expandVocabulary(stats[1]);
					DFs[0] = (int)(Double.parseDouble(stats[3]));
					DFs[1] = (int)(Double.parseDouble(stats[4]));
					setVocabStat(stats[1], DFs);
					
					ngram = 1+Utils.countOccurrencesOf(stats[1], "-");
					if (m_Ngram<ngram)
						m_Ngram = ngram;
				}
			}
			reader.close();
			
			System.out.format("Load %d %d-gram features from %s...\n", m_featureNames.size(), m_Ngram, filename);
			m_isCVLoaded = true;
			m_isCVStatLoaded = true;
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}
	
	void setVocabStat(String term, int[] DFs) {
		_stat stat = m_featureStat.get(term);
		stat.setDF(DFs);
	}
	
	//Load all the users.
	public void loadUserDir(String folder){
		int count = 0;
		if(folder == null || folder.isEmpty())
			return;
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			if(f.isFile()){
				loadOneUser(f.getAbsolutePath());
				count++;
			} else if (f.isDirectory())
				loadUserDir(f.getAbsolutePath());
		}
		System.out.format("%d users are loaded from %s...\n", count, folder);
	}
	
	// Load one file as a user here. 
	public void loadOneUser(String filename){
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			String[] names = filename.split("/");
			int endIndex = names[names.length-1].lastIndexOf(".");
			String userID = names[names.length-1].substring(0, endIndex); //UserId is contained in the filename.
			// Skip the first line since it is user name.
			reader.readLine(); 
			
			String reviewID, source, category;
			ArrayList<_Review> reviews = new ArrayList<_Review>();
			_Review review;
			int ylabel;
			long timestamp;
			while((line = reader.readLine()) != null){
				reviewID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());
				
				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, reviewID, category, timestamp);
					if(AnalyzeDoc(review)) //Create the sparse vector for the review.
						reviews.add(review);
				}
			}
			
			if(reviews.size() > 1){//at least one for adaptation and one for testing
				allocateReviews(reviews);				
				m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//[0, train) for training purpose
	//[train, adapt) for adaptation purpose
	//[adapt, 1] for testing purpose
	void allocateReviews(ArrayList<_Review> reviews) {
		Collections.sort(reviews);// sort the reviews by timestamp
		int train = (int)(reviews.size() * m_trainRatio), adapt = (int)(reviews.size() * (m_trainRatio + m_adaptRatio));
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

	//Return all the users.
	public ArrayList<_User> getUsers(){
		System.out.format("[Info]Training size: %d, adaptation size: %d, and testing size: %d\n", m_trainSize, m_adaptSize,m_testSize);
		return m_users;
	}
}
