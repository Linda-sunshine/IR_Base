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

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.Post;
import structures.Product;
import structures._Doc;
import structures.medicalPost;
import utils.Utils;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class medforumAnalyzer extends DocAnalyzer{
	
	SimpleDateFormat m_dateFormatter;
	
	//Constructor with ngram and fValue.
	public medforumAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
	}
	
	//Constructor with ngram and fValue.
	public medforumAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel, String posModel) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, posModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
	}
	
	//Constructor with ngram and fValue.
	public medforumAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
//		m_dateFormatter = new SimpleDateFormat("yyyy-MM-dddd");// standard date format for yelp data
	}
	//Load a document and analyze it.
	
	String m_category;
	
	public void setCategory(String category){
		this.m_category = category;
	}
	
	@Override
	public void LoadDoc(String filename) {
		medicalPost meds = null;
		JSONArray jarray = null;
		
		try {
			//System.out.println(filename);
			JSONObject json = LoadJson(filename);
			jarray = json.getJSONArray("thread");
		} catch (Exception e) {
			System.err.print("Array Not Found");
			e.printStackTrace();
			return;
		}	
		
		for(int i=0; i<jarray.length(); i++) {
			try {
				meds = new medicalPost(jarray.getJSONObject(i));
				String content = meds.getContent();
				if(this.m_category.equalsIgnoreCase("Depression"))
					meds.setLabel(1);
				else
					meds.setLabel(0);
				_Doc review = new _Doc (m_corpus.getSize(), content, meds.getLabel());
					if(this.m_stnDetector!=null && !m_classifierOrTopicmodel)
						AnalyzeDocWithStnSplit(review);
					else{
							AnalyzeDoc(review);
						}
				}
			catch (JSONException e) {
				System.out.print('P');
			}
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
