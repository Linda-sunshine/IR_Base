//package Analyzer;
//
//import java.io.BufferedReader;
//import java.io.FileInputStream;
//import java.io.FileNotFoundException;
//import java.io.IOException;
//import java.io.InputStreamReader;
//import java.util.ArrayList;
//import java.util.HashMap;
//
//import opennlp.tools.util.InvalidFormatException;
//import structures.TokenizeResult;
//import structures._Doc;
//import structures._Review;
//import structures._SparseFeature;
//import structures._User;
//import structures._stat;
//import utils.Utils;
//
//public class MultiThreadedLMAnalyzer extends MultiThreadedUserAnalyzer {
//	// We don't record the feature stats.
//	ArrayList<String> m_lmFeatureNames;
//	HashMap<String, Integer> m_lmFeatureNameIndex;
//	boolean m_isLMCVLoaded = false;
//
//	public MultiThreadedLMAnalyzer(String tokenModel, int classNo, String providedCV, String lmFvFile,
//			int Ngram, int threshold, int numberOfCores, boolean b)
//			throws InvalidFormatException, FileNotFoundException, IOException {
//		super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
//		m_lmFeatureNames = new ArrayList<String>();
//		m_lmFeatureNameIndex = new HashMap<String, Integer>();
//		loadLMFeatures(lmFvFile);
//	}
//
//	// Added by Lin. Load the features for language models from a file and store them in the m_LMFeatureNames.
//	public boolean loadLMFeatures(String filename){
//		//If no lm features provided, we will use the same features as logistic model.
//		if (filename==null || filename.isEmpty()){
//			m_lmFeatureNameIndex = m_featureNameIndex;
//			m_lmFeatureNames = m_featureNames;
//			System.out.println("Language models share the same features with classification models!");
//			m_isLMCVLoaded = true;
//			return true;
//		}
//
//		try {
//			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
//			String line;
//			while ((line = reader.readLine()) != null) {
//				if (line.startsWith("#"))//comments
//					continue;
//				else{
//					m_lmFeatureNameIndex.put(line, m_lmFeatureNames.size()); // set the index of the new feature.
//					m_lmFeatureNames.add(line); // Add the new feature.
//				}
//			}
//			reader.close();
//			System.out.format("%d features for language model are loaded from %s...\n", m_lmFeatureNames.size(), filename);
//			m_isLMCVLoaded = true;
//			return true;
//
//		} catch (IOException e) {
//			System.err.format("[Error]Failed to open file %s!!", filename);
//			return false;
//		}
//	}
//
//	@Override
//	// Analyze the sparse features and language model features at the same time.
//	protected boolean AnalyzeDoc(_Doc doc, int core) {
//
//		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource(),core);// Three-step analysis.
//		String[] tokens = result.getTokens();
//		int y = doc.getYLabel();
//
//		// Construct the sparse vector.
//		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
//		// Construct the sparse vector for the language models.
//		HashMap<Integer, Double> lmSpVct = constructLMSpVct(tokens);
//
//		if (spVct.size()>m_lengthThreshold) {//temporary code for debugging purpose
//			doc.createSpVct(spVct);
//			doc.createLMSpVct(lmSpVct);
//			doc.setStopwordProportion(result.getStopwordProportion());
//			synchronized (m_corpusLock) {
//				m_corpus.addDoc(doc);
//				m_classMemberNo[y]++;
//			}
//			if (m_releaseContent)
//				doc.clearSource();
//
//			return true;
//		} else {
//			/****Roll back here!!******/
//			synchronized (m_rollbackLock) {
//				rollBack(spVct, y);// no need to roll back lm features.
//			}
//			return false;
//		}
//	}
//
//	//Added by Lin for constructing language model vectors and we don't record stats for features used in LM.
//	public HashMap<Integer, Double> constructLMSpVct(String[] tokens){
//		int lmIndex = 0;
//		double lmValue = 0;
//		HashMap<Integer, Double> lmVct = new HashMap<Integer, Double>();//Collect the index and counts of projected features.
//
//		// We assume we always have the features loaded beforehand.
//		for(int i = 0; i < tokens.length; i++){
//			if (isLegit(tokens[i])){
//				if(m_lmFeatureNameIndex.containsKey(tokens[i])){
//					lmIndex = m_lmFeatureNameIndex.get(tokens[i]);
//					if(lmVct.containsKey(lmIndex)){
//						lmValue = lmVct.get(lmIndex) + 1;
//						lmVct.put(lmIndex, lmValue);
//					} else
//						lmVct.put(lmIndex, 1.0);
//				}
//			}
//		}
//		return lmVct;
//	}
//
//	// Estimate a global language model.
//	// We traverse all review documents instead of using the global TF
//	public double[] estimateGlobalLM(){
//		double[] lm = new double[getLMFeatureSize()];
//		double sum = 0;
//		for(_User u: m_users){
//			for(_Review r: u.getReviews()){
//				for(_SparseFeature fv: r.getLMSparse()){
//					lm[fv.getIndex()] += fv.getValue();
//					sum += fv.getValue();
//				}
//			}
//		}
//		for(int i=0; i<lm.length; i++){
//				lm[i] /= sum;
//				if(lm[i] == 0)
//					lm[i] = 0.0001;
//		}
//			return lm;
//	}
//
//	public int getLMFeatureSize(){
//		return m_lmFeatureNames.size();
//	}
//}
