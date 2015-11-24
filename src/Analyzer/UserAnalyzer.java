package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;

import opennlp.tools.util.InvalidFormatException;
import structures.TokenizeResult;
import structures._Review;
import structures._User;

public class UserAnalyzer extends DocAnalyzer {

	ArrayList<_User> m_users; // Store all users with their reviews.
	int m_count;
	TreeMap<Integer, ArrayList<Integer>> m_featureGroupIndex;
	public UserAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_users = new ArrayList<_User>();
		m_count = 0;
		m_featureGroupIndex = new TreeMap<Integer, ArrayList<Integer>>();
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
			}
			else 
				loadUserDir(f.getAbsolutePath());
		}
		System.out.format("%d users are loaded from %s.", count, folder);
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
				source = reader.readLine();
				category = reader.readLine();
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());
				
				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_count++, source, ylabel, userID, reviewID, category, timestamp);
					AnalyzeDoc(review); //Create the sparse vector for the review.
					reviews.add(review);
				}
			}
			if(reviews.size() != 0)
				m_users.add(new _User(userID, reviews)); //create new user from the file.
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void AnalyzeDoc(_Review doc){
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();
		
		// Construct the sparse vector.
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			doc.setStopwordProportion(result.getStopwordProportion());
//			m_corpus.addDoc(doc);
//			m_classMemberNo[y]++;
			if (m_releaseContent)
				doc.clearSource();
		} else {
			/****Roll back here!!******/
			rollBack(spVct, y);
		}
	}
	
//	public void loadFeatureGroups(String filename){
//		try{
//			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
//			String line = reader.readLine(); //There is only one line.
//			String[] groupNos = line.split(",");
//			int groupNo = 0;
//			for(int i=0; i<groupNos.length; i++){
//				groupNo = Integer.valueOf(groupNos[i]);
//				
//				if(!m_featureGroupIndex.containsKey(groupNo))
//					m_featureGroupIndex.put(groupNo, new ArrayList<Integer>());//If this group hasn't shown up.
//				m_featureGroupIndex.get(groupNo).add(i);
//			}
//			reader.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
	
	/***When we do feature selection, we will group features and store them in file.
	 ***We load the feature group file and store it in hashtable for reference.
	 ***/
	public void fillFeatureGroups(String filename){
		try{
			int groupNo = 0;
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String[] features = reader.readLine().split(",");//Group information of each feature.
			reader.close();
			
			//Analyze the features and corresponding groups, the format is as follows:<Group index, <feature index>>
			m_featureGroupIndex = new TreeMap<Integer, ArrayList<Integer>>();
			
			//Put the first group in the  hashmap.
			m_featureGroupIndex.put(0, new ArrayList<Integer>());
			m_featureGroupIndex.get(0).add(0);//The key of the first group is 0 and the group member is also 0. 
			for(int i=0; i <features.length; i++){
				groupNo = Integer.valueOf(features[i]) + 1;//group index starts from 1, 0th element serves as bias.
				if(!m_featureGroupIndex.containsKey(groupNo))//Create a new arraylist if the key does not exist.
					m_featureGroupIndex.put(groupNo, new ArrayList<Integer>());
				m_featureGroupIndex.get(groupNo).add(i+1);//Feature index starts from 1, 2....
			}
		} catch(IOException e){
			System.err.format("Fail to open file %s.\n", filename);
		}
	}
	
	//Load global model from file.
	public double[] loadGlobalWeights(String filename){
		double[] weights = null;
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String[] features = reader.readLine().split(",");//Group information of each feature.
			reader.close();
			
			weights = new double[features.length];
			for(int i=0; i<features.length; i++)
				weights[i] = Double.valueOf(features[i]);
			
		} catch(IOException e){
			System.err.format("Fail to open file %s.\n", filename);
		}
		return weights;
	}
	
	//Return all the users.
	public ArrayList<_User> getUsers(){
		return m_users;
	}
	
	//How many feature groups are there.
	public int getFeatureGroupNo(){
		return 	m_featureGroupIndex.size();
	}
	
	//Return the feature group index and corresponding features.
	public TreeMap<Integer, ArrayList<Integer>> getFeatureGroupIndex(){
		return m_featureGroupIndex;
	}
}
