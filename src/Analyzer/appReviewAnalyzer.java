/**
 * 
 */
package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.AppReviewPost;
import structures.NewEggPost;
import structures.Post;
import structures.Product;
import structures.TokenizeResult;
import structures._Doc;
import structures._Stn;
import utils.Utils;

/**
 * @author Md Mustafizur Rahman
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class appReviewAnalyzer extends DocAnalyzer{
	
	SimpleDateFormat m_dateFormatter;
	HashMap<String, Integer> annotation = new HashMap<String, Integer>();
	
	//Constructor with ngram and fValue.
	public appReviewAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
	}
	
	//Constructor with ngram and fValue.
	public appReviewAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel, String posModel) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, posModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
	}
	
	//Constructor with ngram and fValue.
	public appReviewAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
//		m_dateFormatter = new SimpleDateFormat("yyyy-MM-dddd");// standard date format for yelp data
	}
	
	
	public void readAnnotation(String fileName){
	
		BufferedReader fileReader = null;
			
		try {
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			while ((line = fileReader.readLine()) != null) {
				String infos[] = line.split("\t");
				String reviewID = infos[0].trim();
				//System.out.println(reviewID);
				int annotationLabel = Integer.parseInt(infos[1]);
				if(!annotation.containsKey(reviewID))
					annotation.put(reviewID, annotationLabel);
				else
					System.out.println("Duplicate Review ID: "+reviewID);
			}
			
			System.out.println("Number of Annotated file:" + annotation.size());

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 



	}
	
	
	protected boolean AnalyzeAppReviewWithSentence(AppReviewPost post, String reviewID, int label) throws ParseException {
		String content;
		TokenizeResult result;
		ArrayList<_Stn> stnList = new ArrayList<_Stn>(); // to avoid empty sentences
		ArrayList<HashMap<Integer, Double>> spVcts = new ArrayList<HashMap<Integer, Double>>(); // Collect the index and counts of features.
		StringBuffer buffer = m_releaseContent?null:new StringBuffer(256);
		HashMap<Integer, Double> vPtr, docVct = new HashMap<Integer, Double>(); // docVct is used to collect DF
		int y = label, uniWordsInSections = 0;
		
		

		if ((content=post.getContent()) != null) {// tokenize 
			for(String sentence : m_stnDetector.sentDetect(content)) {
				result = TokenizerNormalizeStemmer(sentence);
				vPtr = constructSpVct(result.getTokens(), y, docVct);

				if (vPtr.size()>0) {//avoid empty sentence
					String[] posTags = m_tagger.tag(result.getRawTokens()); // only tokenize then POS tagging

					stnList.add(new _Stn(Utils.createSpVct(vPtr), result.getRawTokens(), posTags, sentence, label));
					uniWordsInSections += vPtr.size();
					Utils.mergeVectors(vPtr, docVct);
					spVcts.add(vPtr);
				}
			}
			if (!m_releaseContent)
				buffer.append(String.format("Content: %s\n", content));
		}
		if (uniWordsInSections>=m_lengthThreshold && stnList.size()>=m_stnSizeThreshold) {
			_Doc doc = new _Doc(m_corpus.getSize(), reviewID, post.getAppId(), post.getAppName(), post.getTitle(), content, post.getVersion(), label);
			doc.setSourceType(3); // 3 means has app data
			doc.createSpVct(spVcts);
			doc.setYLabel(y);
			doc.setSentences(stnList);
			setStnFvs(doc);
			m_corpus.addDoc(doc);
			m_classMemberNo[y]++;
			return true;
		} else
			return false;
	}

	//Load a document and analyze it.
	@Override
	public void LoadDoc(String filename) {
		JSONObject json = null;
		
		try {
			json = LoadJson(filename);
			AppReviewPost post = new AppReviewPost(json);
			
			String content;
			if (Utils.endWithPunct(post.getTitle()))
				content = post.getTitle() + " " + post.getContent();
			else
				content = post.getTitle() + ". " + post.getContent();

			//public _Doc (int ID, String reviewID, String appID, String appName, String title, String source, String version, int rating){
			
			String reviewID = post.getID();
			int annotationLabel = -1;
			if(annotation.containsKey(reviewID))
			{
				annotationLabel = annotation.get(reviewID);
			}
			else{
				System.out.println("Annotation unavailable for reviewID: "+reviewID);
				annotationLabel = 0;
			}
			_Doc review = new _Doc(m_corpus.getSize(), reviewID, post.getAppId(), post.getAppName(), post.getTitle(), content, post.getVersion(), annotationLabel);
			review.setSourceType(3); // 3 means has app data
			
			if(this.m_stnDetector!=null && !m_classifierOrTopicmodel)
				AnalyzeAppReviewWithSentence(post, reviewID, annotationLabel);
			else if(!m_classifierOrTopicmodel) // if false
				AnalyzeDoc(review);
		} 
		catch (Exception e) {
			System.out.println("Cannot load "+filename);
			e.printStackTrace();
			return;
		}	
	}
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
}
