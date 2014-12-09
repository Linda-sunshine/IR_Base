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
import structures._Doc;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class jsonAnalyzer extends DocAnalyzer{
	
	private SimpleDateFormat m_dateFormatter;
	
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, String fs) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, fs);		
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");//standard date format for this project
	}	
	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, String fs, int Ngram) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, fs, Ngram);		
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");//standard date format for this project
	}
	
	//Load a document and analyze it.
	@Override
	public void LoadDoc(String filename) {
		try {
			JSONObject json = LoadJson(filename);
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post post = new Post(jarray.getJSONObject(i));
				if (checkPostFormat(post)){
					long timeStamp = this.m_dateFormatter.parse(post.getDate()).getTime();
					AnalyzeDoc(new _Doc(m_corpus.getSize(), post.getContent(), (post.getLabel()-1), timeStamp));
					this.m_classMemberNo[post.getLabel()-1]++;
				} else{
					System.err.format("*******Wrong review!! Ignored!!******\n");
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		}
	}
	
	//sample code for loading the json file
	public JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				//System.out.println(line);
				buffer.append(line);
			}
			reader.close();
			return new JSONObject(buffer.toString());
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!", filename);
			e.printStackTrace();
			return null;
		} catch (JSONException e) {
			System.err.format("[Error]Failed to parse json file %s!", filename);
			e.printStackTrace();
			return null;
		}
	}
	
	//check format for each post
	private boolean checkPostFormat(Post p) {
		if (p.getLabel() <= 0 || p.getLabel() > 5){
			System.err.format("[Error]Missing Lable or wrong label!!");
			return false;
		}
		else if (p.getContent() == null){
			System.err.format("[Error]Missing content!!\n");
			return false;
		}	
		else if (p.getDate() == null){
			System.err.format("[Error]Missing date!!\n");
			return false;
		}
		else {
			// to check if the date format is correct
			try {
				m_dateFormatter.parse(p.getDate());
				//System.out.println(p.getDate());
				return true;
			} catch (ParseException e) {
				System.err.format("[Error]Wrong date format!", p.getDate());
			}
			return true;
		} 
	}
}
