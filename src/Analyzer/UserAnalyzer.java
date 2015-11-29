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
	
	int m_count;
	ArrayList<_User> m_users; // Store all users with their reviews.
	int[] m_featureGroupIndexes; //The array of feature group indexes.
	
	public UserAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_users = new ArrayList<_User>();
		m_count = 0;
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
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, reviewID, category, timestamp);
					if(AnalyzeDoc(review)) //Create the sparse vector for the review.
						reviews.add(review);
				}
			}
			if(reviews.size() != 0){
				m_users.add(new _User(userID, reviews)); //create new user from the file.
				m_corpus.addDocs(reviews);
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public boolean AnalyzeDoc(_Review doc){
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();
		
		// Construct the sparse vector.
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			doc.setStopwordProportion(result.getStopwordProportion());

			if (m_releaseContent)
				doc.clearSource();
			return true;
		} else {
			/****Roll back here!!******/
			rollBack(spVct, y);
			return false;
		}
	}
	
	/***When we do feature selection, we will group features and store them in file. 
	 * The index is the index of features and the corresponding number is the group index number.***/
	public void loadFeatureGroupIndexes(String filename){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String[] features = reader.readLine().split(",");//Group information of each feature.
			reader.close();
			
			m_featureGroupIndexes = new int[features.length + 1]; //One more term for bias, bias=0.
			//Group index starts from 0, so add 1 for it.
			for(int i=0; i<features.length; i++)
				m_featureGroupIndexes[i+1] = Integer.valueOf(features[i]) + 1;
			
		} catch(IOException e){
			System.err.format("Fail to open file %s.\n", filename);
		}
	}
	
	public int[] getFeatureGroupIndexes(){
		return m_featureGroupIndexes;
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
}
