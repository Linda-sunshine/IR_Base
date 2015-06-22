/**
 * 
 */
package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.HashMap;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.Post;
import structures.Product;
import structures._Doc;
import utils.Utils;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class jsonAnalyzer extends DocAnalyzer{
	
	SimpleDateFormat m_dateFormatter;
	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
	}
	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
	}
	//previous LoadDoc, in case we need it.
	public void LoadDoc(String filename) {
		Product prod = null;
		JSONArray jarray = null;
		
		try {
			JSONObject json = LoadJson(filename);
			prod = new Product(json.getJSONObject("ProductInfo"));
			jarray = json.getJSONArray("Reviews");
		} catch (Exception e) {
			System.out.print('X');
			return;
		}	
		
		for(int i=0; i<jarray.length(); i++) {
			try {
				Post post = new Post(jarray.getJSONObject(i));
				if (checkPostFormat(post)){
					long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
					String content;
					if (Utils.endWithPunct(post.getTitle()))
						content = post.getTitle() + " " + post.getContent();
					else
						content = post.getTitle() + ". " + post.getContent();

					int label = 0;
					if(post.getLabel() >= 4) label = 1;
					_Doc review = new _Doc(m_corpus.getSize(), post.getID(), content, prod.getID(), label, timeStamp);
//					_Doc review = new _Doc(m_corpus.getSize(), post.getID(), content, prod.getID(), post.getLabel()-1, timeStamp);
					if(this.m_stnDetector!=null)
//						AnnotateIndex(review);
						AnalyzeDocWithStnSplit(review);
					else
						AnalyzeDoc(review);
				}
			} catch (ParseException e) {
				System.out.print('T');
			} catch (JSONException e) {
				System.out.print('P');
			}
		}
	}
	
	/***Load a json file with 1:1 postive and negative reviews.
	Will this ruin the bias given by the ratio since this ratio is informative itself.***/
//	public void LoadDoc(String filename) {
//		Product prod = null;
//		JSONArray jarray = null;
//		int[] count = new int[m_classNo];
//		
//		try {
//			JSONObject json = LoadJson(filename);
//			prod = new Product(json.getJSONObject("ProductInfo"));
//			jarray = json.getJSONArray("Reviews");
//		} catch (Exception e) {
//			System.out.print('X');
//			return;
//		}	
//		//Get the minimum number of reviews
//		for(int i = 0; i < jarray.length(); i++){
//			try {
//				Post post = new Post(jarray.getJSONObject(i));
//				if (checkPostFormat(post)){
//					int label = 0;
//					if(post.getLabel() >= 3) label = 1;
//					count[label]++;
//				}
//			} catch (JSONException e) {
//				System.out.print('P');
//			}
//		}
//		int min = Utils.minOfArrayValue(count);
//		count = new int[m_classNo];
//		for(int i=0; i<jarray.length(); i++) {
//			try {
//				Post post = new Post(jarray.getJSONObject(i));
//				if (checkPostFormat(post)){
//					long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
//					String content;
//					if (Utils.endWithPunct(post.getTitle()))
//						content = post.getTitle() + " " + post.getContent();
//					else
//						content = post.getTitle() + ". " + post.getContent();
//
//					int label = 0;
//					if(post.getLabel() >= 3) label = 1;
//					if(count[label] < min){
//						_Doc review = new _Doc(m_corpus.getSize(), post.getID(), content, prod.getID(), label, timeStamp);
//						count[label]++;
////						_Doc review = new _Doc(m_corpus.getSize(), post.getID(), content, prod.getID(), post.getLabel()-1, timeStamp);
//						if(this.m_stnDetector!=null)
//							AnalyzeDocWithStnSplit(review);
//						else
//							AnalyzeDoc(review);
//					}
//				}
//			} catch (ParseException e) {
//				System.out.print('T');
//			} catch (JSONException e) {
//				System.out.print('P');
//			}
//		}
//	}
	
	//sample code for loading the json file
	JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				buffer.append(line);
			}
			reader.close();
			return new JSONObject(buffer.toString());
		} catch (Exception e) {
			System.out.print('X');
			return null;
		}
	}
	
	//check format for each post
	private boolean checkPostFormat(Post p) {
		if (p.getLabel() <= 0 || p.getLabel() > 5){
			//System.err.format("[Error]Missing Lable or wrong label!!");
			System.out.print('L');
			return false;
		}
		else if (p.getContent() == null){
			//System.err.format("[Error]Missing content!!\n");
			System.out.print('C');
			return false;
		}	
		else if (p.getDate() == null){
			//System.err.format("[Error]Missing date!!\n");
			System.out.print('d');
			return false;
		}
		else {
			// to check if the date format is correct
			try {
				m_dateFormatter.parse(p.getDate());
				//System.out.println(p.getDate());
				return true;
			} catch (ParseException e) {
				System.out.print('D');
			}
			return true;
		} 
	}
	
//	protected boolean AnalyzeDoc(_Doc doc) {
//		if(doc.getYLabel() == 1 && m_classMemberNo[1] >= 3185)
//			return true;
//		else{
//			String[] tokens = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
//			HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
//			int index = 0;
//			double value = 0;
//			// Construct the sparse vector.
//			for (String token : tokens) {
//				// CV is not loaded, take all the tokens as features.
//				if (!m_isCVLoaded) {
//					if (m_featureNameIndex.containsKey(token)) {
//						index = m_featureNameIndex.get(token);
//						if (spVct.containsKey(index)) {
//							value = spVct.get(index) + 1;
//							spVct.put(index, value);
//						} else {
//							spVct.put(index, 1.0);
//							m_featureStat.get(token).addOneDF(doc.getYLabel());
//						}
//					} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
//						expandVocabulary(token);// update the m_featureNames.
//						index = m_featureNameIndex.get(token);
//						spVct.put(index, 1.0);
//						m_featureStat.get(token).addOneDF(doc.getYLabel());
//					}
//					m_featureStat.get(token).addOneTTF(doc.getYLabel());
//				} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
//					index = m_featureNameIndex.get(token);
//					if (spVct.containsKey(index)) {
//						value = spVct.get(index) + 1;
//						spVct.put(index, value);
//					} else {
//						spVct.put(index, 1.0);
//						m_featureStat.get(token).addOneDF(doc.getYLabel());
//					}
//					m_featureStat.get(token).addOneTTF(doc.getYLabel());
//				}
//				// if the token is not in the vocabulary, nothing to do.
//			}
//			if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose
//				doc.createSpVct(spVct);
//				m_corpus.addDoc(doc);
//				m_classMemberNo[doc.getYLabel()]++;
//			}
//		}
//		return true;
////		if (m_releaseContent){
////			doc.clearSource();
////			return true;
////		}
////		else
////			return false;
//	}
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		String folder = "./data/amazon/small";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model
		String fvFile = "./data/Features/fv_2gram_3185.txt";
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, 2, fvFile, 2, 10);
		analyzer.LoadDirectory(folder, suffix);
	}
}
