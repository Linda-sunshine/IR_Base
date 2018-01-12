package Application;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedLMAnalyzer;

public class LinkPredPreprocess {
	ArrayList<_User> m_users;
	HashMap<String, Integer> m_userIDMap = new HashMap<String, Integer>();
	Set<String> m_trainUserIDs = new HashSet<String>();
	Set<String> m_testUserIDs = new HashSet<String>();
	
	public LinkPredPreprocess(ArrayList<_User> users){
		// build the look-up table
		for(int i=0; i<users.size(); i++){
			_User user = users.get(i);
			m_userIDMap.put(user.getUserID(), i);
			user.calculateTrainTestReview();
			if(user.getTrainReviewSize() == 0)
				m_testUserIDs.add(user.getUserID());
			else if(user.getTestReviewSize() == 0)
				m_trainUserIDs.add(user.getUserID());
		}
		m_users = users;
	}
	// save the user-user pair to graphlab/svd for model training.
	public void saveUsers4SVD(ArrayList<_User> users){
		
		
	}	
	
	// save the user-user pairs to graphlab for model training
	public void saveUserUserPairs4FM(String dir, int trainSize, int testSize){
		int trainPair = 0, testPair = 0;

		try{
			PrintWriter trainWriter = new PrintWriter(new File(String.format("%s/train_%d_test_%d_train_user_rcmd.csv", dir, trainSize, testSize)));
			PrintWriter testWriter = new PrintWriter(new File(String.format("%s/train_%d_test_%d_test_user_rcmd.csv", dir, trainSize, testSize)));
			trainWriter.write("user_id,item_id,rating\n");
			testWriter.write("user_id,item_id,rating\n");
			for(int i=0; i<m_users.size(); i++){
				_User user = m_users.get(i);
				// if it is a train user, can only access train users as friends
				if(user.getTestReviewSize() == 0){
					for(String frd: user.getFriends()){
						if(m_trainUserIDs.contains(frd)){
							trainWriter.write(String.format("%s,%s,%d\n", user.getUserID(), frd, 1));
							trainPair++;
						}
					}
				// if it is a test user, can access all the users as friends
				} else if(user.getTrainReviewSize() == 0){
					for(String uid: m_trainUserIDs){
						if(user.hasFriend(uid)){
							testWriter.write(String.format("%s,%s,%d\n", user.getUserID(), uid, 1));
						} else{
							testWriter.write(String.format("%s,%s,%d\n", user.getUserID(), uid, 0));
						}
						testPair++;
					}
					// accessing testing user ids
					for(String uid: m_testUserIDs){
						if(user.hasFriend(uid)){
							testWriter.write(String.format("%s,%s,%d\n", user.getUserID(), uid, 1));
						} else{
							testWriter.write(String.format("%s,%s,%d\n", user.getUserID(), uid, 0));
						}
						testPair++;
					}
				}
			}
			trainWriter.close();
			testWriter.close();
			System.out.format("[Info]Finish writing (%d,%d) training users/pairs, (%d,%d) testing users/pairs.\n", 
					m_trainUserIDs.size(), trainPair, m_testUserIDs.size(), testPair);
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		boolean enforceAdapt = true;
		String dataset = "YelpNew"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		int lmTopK = 1000; // topK for language model.
		String fs = "DF";//"IG_CHI"
		String prefix = "./data/CoLinAdapt";
		String userPrefix = "/home/lin/DataSigir/YelpNew";

		for(int i=2; i<=8; i++){
			int trainSize = i*1000;	
			int testSize = 2000;
			String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
			String trainFolder = String.format("%s/Users_%d_train", userPrefix, trainSize);
			String testFolder =  String.format("%s/Users_%d_test", userPrefix, testSize);
			String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
			if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
			String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
			MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
			adaptRatio = 1; enforceAdapt = true;
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		
			// load training users with (adaptRatio=1, testRatio=0)
			analyzer.loadUserDir(trainFolder);
		
			// load testing users with (adaptaRatio=0, testRatio=1)
			adaptRatio = 0; enforceAdapt = false;
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.loadUserDir(testFolder);
		
			analyzer.buildFriendship(friendFile);
			analyzer.setFeatureValues("TFIDF-sublinear", 0);	
		
			LinkPredPreprocess lpp = new LinkPredPreprocess(analyzer.getUsers());
			lpp.saveUserUserPairs4FM(userPrefix+"/linkPredFM", trainSize, testSize);
			System.out.println("------------------------------------------------");
		}
		
	}
}
