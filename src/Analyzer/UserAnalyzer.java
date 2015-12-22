package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeMap;

import opennlp.tools.util.InvalidFormatException;
import structures.TokenizeResult;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._stat;
import utils.Utils;

public class UserAnalyzer extends DocAnalyzer {
	
	int m_count;
	ArrayList<_User> m_users; // Store all users with their reviews.
	int[] m_featureGroupIndexes; // The array of feature group indexes.
	double[] m_DFs; // The array stores the total DF for all features.
	ArrayList<Double> m_globalWeights;
	
	public UserAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_users = new ArrayList<_User>();
		m_count = 0;
		m_globalWeights = new ArrayList<Double>();
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
				if(count%100 == 0)
					System.out.print(".");
				count++;
			}
			else 
				loadUserDir(f.getAbsolutePath());
		}
		System.out.format("\n%d users are loaded from %s.", count, folder);
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
				m_users.add(new _User(userID, reviews, m_users.size())); //create new user from the file.
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
	
	//This file only contains the weights for global model.
//	public double[] loadGlobalWeights(String filename){
//		double[] weights = null;
//		try{
//			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
//			String[] features = reader.readLine().split(",");//Group information of each feature.
//			reader.close();
//			
//			weights = new double[features.length];
//			for(int i=0; i<features.length; i++)
//				weights[i] = Double.valueOf(features[i]);
//			
//		} catch(IOException e){
//			System.err.format("Fail to open file %s.\n", filename);
//		}
//		return weights;
//	}
	
	//Load global model from file, each line is as follows: feature, feature weight.
	public void loadGlobalWeights(String filename){
		try{
			String line, feature = null;
			double weight = 0;
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String[] featureWeight;
			
			while((line = reader.readLine()) != null){
				featureWeight = line.split(":");//Group information of each feature.
				if(featureWeight.length == 2)
//					feature = featureWeight[0];
					weight = Double.valueOf(featureWeight[1]);
//					if(!feature.equals("bais"))
//						m_featureNames.add(feature);//Add the feature to the feature list.
					m_globalWeights.add(weight);//Add the global weight.
			}
			System.out.format("%d weigths are loaded.\n", m_globalWeights.size());
			reader.close();
		} catch(IOException e){
			System.err.format("Fail to open file %s.\n", filename);
		}
	}
	
	//Transfer the global weights to array and return it.
	public double[] getGlobalWeights(){
		double[] weights = new double[m_globalWeights.size()];
		for(int i=0; i<m_globalWeights.size(); i++)
			weights[i] = m_globalWeights.get(i);
		return weights;
	}
	//Return all the users.
	public ArrayList<_User> getUsers(){
		return m_users;
	}
	
	//Overwrite the LoadCV function since we need the stat of the features from the training data of global model.
                   
	public boolean LoadCVStat(String filename){
		if(filename==null || filename.isEmpty())
			return false;
		try{
			String line, feature;
			String[] fvStat;
			int index;
			double DF;
//			m_features = new String[5000];
			m_DFs = new double[5000]; //Assign the space beforehand.
			
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			while((line = reader.readLine()) != null){
				fvStat = line.split(",");
				//Split the string and get the DF of one feature.
				if(fvStat.length == 5){
					feature = fvStat[1];
					index=super.m_featureNames.indexOf(feature);
					DF = Double.valueOf(fvStat[2]);
//					m_features[index] = feature;
					m_DFs[index] = DF;
//					m_featureNameIndex.put(feature, index);
//					m_featureStat.put(feature, new _stat(m_classNo));
				}
			}
			reader.close();
//			m_featureNames = new ArrayList<String>(Arrays.asList(m_features));
			System.out.format("%d feature stat is loaded.\n", m_DFs.length);
			return true;
		}
		catch(IOException e){
			e.printStackTrace();
			return false;
		}
	}
	
	//Overwrite the setting function with stat from global data set.
	public void setFeatureValues(int N, String fValue, int norm){
		ArrayList<_Doc> docs = m_corpus.getCollection(); // Get the collection of all the documents in the corpus.
		if (fValue.equals("TF")){
			//the original feature is raw TF
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					double DF = m_DFs[sf.getIndex()];
					double IDF = Math.log((N + 1) / DF); //The N is the total number of docs in training data.
					avgIDF += IDF;
				}
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("TFIDF")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					double TF =  Math.log10(sf.getValue())+ 1 ;// normalized TF
					double DF = m_DFs[sf.getIndex()];
					double IDF = Math.log10((25265)/DF)+ 1;
//					double TF = sf.getValue() / temp.getTotalDocLength();// normalized TF
//					double TF = sf.getValue();
//					double DF = m_DFs[sf.getIndex()];
//					double IDF = Math.log((N + 1) / DF);
					double TFIDF = TF * IDF;
					sf.setValue(TFIDF);
					avgIDF += IDF;
//					(1+Math.log10(r.m_VSM.get(key)))*(1+Math.log10(Config.NumberOfReviewsInTraining/m_Vocabs.get(key).getValue())));
				}
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("BM25")) {
			double k1 = 1.5; // [1.2, 2]
			double b = 0.75; // (0, 1000]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg, avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N - DF + 0.5) / (DF + 0.5));
					double BM25 = IDF * TF * (k1 + 1) / (k1 * (1 - b + b * n) + TF);
					sf.setValue(BM25);
					avgIDF += IDF;
				}
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("PLN")) {
			double s = 0.5; // [0, 1]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg, avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N + 1) / DF);
					double PLN = (1 + Math.log(1 + Math.log(TF)) / (1 - s + s * n)) * IDF;
					sf.setValue(PLN);
					avgIDF += IDF;
				}
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else {
			//The default value is just keeping the raw count of every feature.
			System.out.println("No feature value is set, keep the raw count of every feature.");
		}
		
		//rank the documents by product and time in all the cases
		//Collections.sort(m_corpus.getCollection());
		if (norm == 1){
			for(_Doc d:docs)			
				Utils.L1Normalization(d.getSparse());
		} else if(norm == 2){
			for(_Doc d:docs)			
				Utils.L2Normalization(d.getSparse());
		} else {
			System.out.println("No normalizaiton is adopted here or wrong parameters!!");
		}
		
		System.out.format("Text feature generated for %d documents...\n", m_corpus.getSize());
	}
}
