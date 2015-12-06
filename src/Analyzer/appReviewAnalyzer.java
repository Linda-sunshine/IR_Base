/**
 * 
 */
package Analyzer;

import java.io.BufferedReader;
import java.io.File;
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
import structures.annotationType;
import utils.Utils;

/**
 * @author Md Mustafizur Rahman
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class appReviewAnalyzer extends DocAnalyzer{
	
	SimpleDateFormat m_dateFormatter;
	HashMap<String, Integer> annotation = new HashMap<String, Integer>();
	int bugCounter =0;
	int normalCounter = 0;
	String category = "";
	
	public void setCategory(String category){
		this.category = category;
	}
	
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
	
	public void printStat(){
		
		System.out.println("Normal review: "+normalCounter+"\nBug Review: "+bugCounter);
	}
	
	
	public void AnalyzeAppReviewSentenceClassification(AppReviewPost post) throws ParseException {
		
		ArrayList<String> content;
		TokenizeResult result;
		StringBuffer buffer = m_releaseContent?null:new StringBuffer(256);
		HashMap<Integer, Double> vPtr, docVct = new HashMap<Integer, Double>(); // docVct is used to collect DF

		/*Analyzing Content Section*/
		if ((content=post.getSentences()) != null) {// tokenize 
			for(int i=0; i<content.size();i++) {
				result = TokenizerNormalizeStemmer(content.get(i));
				int y = post.getSentenJudgedLabel().get(i);
				//if(y<0) continue;
				vPtr = constructSpVct(result.getTokens(), y<0? 0: y, docVct);
				if (vPtr.size()>=m_lengthThreshold) {
					_Doc doc = new _Doc(m_corpus.getSize(), post.getID(), post.getAppId(), post.getAppName(), post.getTitle(), post.getContent(), post.getVersion());
					if(y<0)
						doc.setAnnotationType(annotationType.UNANNOTATED);
					else
						doc.setAnnotationType(annotationType.PARTIALLY_ANNOTATED); // source = 2 means the Document has judgement
					doc.createSpVct(vPtr);
					doc.setYLabel(y);
					m_corpus.addDoc(doc);
					m_classMemberNo[y<0? 0: y]++;
				}
			}
			if (!m_releaseContent)
				buffer.append(String.format("Content: %s\n", content));
		}
	}
	
	
	public void AnalyzeAppReviewUnannotatedSentenceClassification(AppReviewPost post) throws ParseException {

		ArrayList<String> content;
		TokenizeResult result;
		StringBuffer buffer = m_releaseContent?null:new StringBuffer(256);
		HashMap<Integer, Double> vPtr, docVct = new HashMap<Integer, Double>(); // docVct is used to collect DF

		/*Analyzing Content Section*/
		if ((content=post.getSentences()) != null) {// tokenize 
			for(int i=0; i<content.size();i++) {
				result = TokenizerNormalizeStemmer(content.get(i));
				int y = 0;
				vPtr = constructSpVct(result.getTokens(), y, docVct);
				if (vPtr.size()>=m_lengthThreshold) {
					_Doc doc = new _Doc(m_corpus.getSize(), post.getID(), post.getAppId(), post.getAppName(), post.getTitle(), post.getContent(), post.getVersion());
					//doc.setSourceType(1); // source = 1 means the Document has no judgement
					doc.setAnnotationType(annotationType.UNANNOTATED);
					doc.createSpVct(vPtr);
					doc.setYLabel(y);
					m_corpus.addDoc(doc);
					m_classMemberNo[y]++;
				}
			}
			if (!m_releaseContent)
				buffer.append(String.format("Content: %s\n", content));
		}
	}
	
	
	protected boolean AnalyzeAppReviewWithSentence(AppReviewPost post, String reviewID, annotationType AnnotationType) throws ParseException {
		ArrayList<String> content;
		TokenizeResult result;
		ArrayList<_Stn> stnList = new ArrayList<_Stn>(); // to avoid empty sentences
		ArrayList<HashMap<Integer, Double>> spVcts = new ArrayList<HashMap<Integer, Double>>(); // Collect the index and counts of features.
		StringBuffer buffer = m_releaseContent?null:new StringBuffer(256);
		HashMap<Integer, Double> vPtr, docVct = new HashMap<Integer, Double>(); // docVct is used to collect DF
		int y = 0; int uniWordsInSections = 0;
		
		if ((content=post.getSentences()) != null) {// tokenize 
			for(int i=0; i<content.size();i++) {
				result = TokenizerNormalizeStemmer(content.get(i));
				y = post.getSentenJudgedLabel().get(i);
				//if(y<0 && AnnotationType==annotationType.PARTIALLY_ANNOTATED) continue;
				vPtr = constructSpVct(result.getTokens(), y<0?0:y, docVct);

				if (vPtr.size()>0) {//avoid empty sentence
					String[] posTags = m_tagger.tag(result.getRawTokens()); // only tokenize then POS tagging
					stnList.add(new _Stn(Utils.createSpVct(vPtr), result.getRawTokens(), posTags, content.get(i), y));
					uniWordsInSections += vPtr.size();
					Utils.mergeVectors(vPtr, docVct);
					spVcts.add(vPtr);
				}
			}
			if (!m_releaseContent)
				buffer.append(String.format("Content: %s\n", content));
		}
		if (uniWordsInSections>=m_lengthThreshold && stnList.size()>=m_stnSizeThreshold) {
			_Doc doc = new _Doc(m_corpus.getSize(), reviewID, post.getAppId(), post.getAppName(), post.getTitle(), post.getContent(), post.getVersion());
			doc.setAnnotationType(annotationType.PARTIALLY_ANNOTATED);
			doc.createSpVct(spVcts);
			//doc.setYLabel(y<0?2:y); // because annotation can be -1 but we transformed to 2
			doc.setYLabel(post.getRating()-1); // because annotation can be -1 but we transformed to 2
			doc.setSentences(stnList);
			setStnFvs(doc);
			m_corpus.addDoc(doc);
			m_classMemberNo[post.getRating()-1]++;
			return true;
		} else
			return false;
	}

	
	
	//Load all the files in the directory.
	public void LoadDirectory(String folder, String suffix, boolean annotated) throws IOException {
		if (folder==null || folder.isEmpty())
			return;

		int current = m_corpus.getSize();
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				if(annotated)
					LoadAnnotatedDoc(f.getAbsolutePath());
				else
					LoadUnAnnotatedDoc(f.getAbsolutePath());
			} else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		System.out.format("Loading %d reviews from %s\n", m_corpus.getSize()-current, folder);
	}
	
	
	
	//Load a document and analyze it.

	public void LoadUnAnnotatedDoc(String filename) {
		JSONObject json = null;
		
		try {
			json = LoadJson(filename);
			
			AppReviewPost post = new AppReviewPost(json);
			String content;
			
			content = post.getContent();
			//System.out.println("Filename:"+filename);
			//System.out.println("c:"+ content);
			String reviewCategory = post.getCategory();
			if(!reviewCategory.equalsIgnoreCase(category))
				return;
			for(int i:post.getSentenJudgedLabel())
			{
				if(i==0)
					normalCounter++;
				if(i==1)
					bugCounter++;
			}
			
			String reviewID = post.getID();
			_Doc review = new _Doc(m_corpus.getSize(), reviewID, post.getAppId(), post.getAppName(), post.getTitle(), content, post.getVersion());
			review.setAnnotationType(annotationType.UNANNOTATED); // 1 means has app data has no judgement
			annotationType AnnotationType = annotationType.UNANNOTATED;
			
			if(this.m_stnDetector!=null && !m_classifierOrTopicmodel)
				AnalyzeAppReviewWithSentence(post, reviewID, AnnotationType);
			else{ /*if(!m_classifierOrTopicmodel) // if false
				AnalyzeDoc(review);*/
				if(!m_classifierOrTopicmodel) // if false
					AnalyzeDoc(review);
				else // if true
					AnalyzeAppReviewUnannotatedSentenceClassification(post);
			}
		
		} 
		catch (Exception e) {
			System.out.println("Cannot load "+filename);
			e.printStackTrace();
			return;
		}	
	}
	
	
	public void LoadAnnotatedDoc(String filename) {
		JSONObject json = null;
		
		try {
			json = LoadJson(filename);
			
			AppReviewPost post = new AppReviewPost(json);
			String content;
			
			content = post.getContent();
			//System.out.println("Filename:"+filename);
			//System.out.println("c:"+ content);
			String reviewCategory = post.getCategory();
			if(!reviewCategory.equalsIgnoreCase(category))
				return;
			for(int i:post.getSentenJudgedLabel())
			{
				if(i==0)
					normalCounter++;
				if(i==1)
					bugCounter++;
			}
			
			String reviewID = post.getID();
			_Doc review = new _Doc(m_corpus.getSize(), reviewID, post.getAppId(), post.getAppName(), post.getTitle(), content, post.getVersion());
			review.setAnnotationType(annotationType.PARTIALLY_ANNOTATED); // 1 means has app data has no judgement
			annotationType AnnotationType = annotationType.PARTIALLY_ANNOTATED;
			
			if(this.m_stnDetector!=null && !m_classifierOrTopicmodel)
				AnalyzeAppReviewWithSentence(post, reviewID, AnnotationType);
			else{ /*if(!m_classifierOrTopicmodel) // if false
				AnalyzeDoc(review);*/
				if(!m_classifierOrTopicmodel) // if false
					AnalyzeDoc(review);
				else // if true
					AnalyzeAppReviewSentenceClassification(post);
			}
		
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
