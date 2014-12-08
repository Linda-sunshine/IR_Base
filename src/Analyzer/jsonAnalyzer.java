/**
 * 
 */
package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import opennlp.tools.util.InvalidFormatException;
import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import structures.Post;
import structures._Doc;
import structures._SparseFeature;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class jsonAnalyzer extends Analyzer{
	
	private int m_window; //The length of the window which means how many labels will be taken into consideration.
	private LinkedList<Integer> m_YLabelQueue;
	private LinkedList<_SparseFeature[]> m_SpVctQueue;
	SimpleDateFormat m_dateFormatter;

	
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, String fs) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		if (!providedCV.equals(""))
			this.LoadCV(providedCV);
		if (!fs.equals("")) {
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}
		this.m_window = 0;
		this.m_YLabelQueue = new LinkedList<Integer>();
		this.m_SpVctQueue = new LinkedList<_SparseFeature[]>();
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");//standard date format for this project
	}	
	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, String fs, int Ngram) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, Ngram);
		if (!providedCV.equals(""))
			this.LoadCV(providedCV);
		if (!fs.equals("")) {
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}
		this.m_window = 0;
		this.m_YLabelQueue = new LinkedList<Integer>();
		this.m_SpVctQueue = new LinkedList<_SparseFeature[]>();
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");//standard date format for this project
	}
	
	//Load all the files in the directory.
	public void LoadDirectory(String folder, String suffix) throws ParseException {
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)){
				AnalyzeThreadedDiscussion(LoadJson(f.getAbsolutePath()));
			}
			else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
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
	
	//Analyze every review, parse it into a document.
	public void AnalyzeThreadedDiscussion(JSONObject json) throws ParseException {		
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post post = new Post(jarray.getJSONObject(i));
				if (checkPostFormat(post)){
					long timeStamp = this.m_dateFormatter.parse(post.getDate()).getTime();
					//System.out.println(post.getLabel());
					AnalyzeDoc(new _Doc(m_corpus.getSize(), post.getContent(), (post.getLabel()-1), timeStamp));
					this.m_classMemberNo[post.getLabel()-1]++;
				} else{
					System.err.format("*******Wrong review!! Ignored!!******\n");
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
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
	
	//Analyze the document as usual.
	public void AnalyzeDoc(_Doc doc) {
		try{
			String[] tokens = TokenizerNormalizeStemmer(doc.getSource());//Three-step analysis.
			doc.setTotalLength(tokens.length); //set the length of the document.
			HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); //Collect the index and counts of features.
			int index = 0;
			double value = 0;
			//Construct the sparse vector.
			for(String token:tokens) {
				//CV is not loaded, take all the tokens as features.
				if(!m_isCVLoaded){
					if (m_featureNameIndex.containsKey(token)) {
						index = m_featureNameIndex.get(token);
						if(spVct.containsKey(index)){
							value = spVct.get(index) + 1;
							spVct.put(index, value);
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						} else{
							spVct.put(index, 1.0);
							this.m_featureStat.get(token).addOneDF(doc.getYLabel());
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						}
					} 
					else{
						//indicate we allow the analyzer to dynamically expand the feature vocabulary
						expandVocabulary(token);//update the m_featureNames.
						updateFeatureStat(token);
						index = m_featureNameIndex.get(token);
						spVct.put(index, 1.0);
						this.m_featureStat.get(token).addOneDF(doc.getYLabel());
						this.m_featureStat.get(token).addOneTTF(doc.getYLabel());					
					}
				//CV is loaded.
				} else if (m_featureNameIndex.containsKey(token)) { 
					index = m_featureNameIndex.get(token);
					if(spVct.containsKey(index)){
						value = spVct.get(index) + 1;
						spVct.put(index, value);
						
					} else {
						spVct.put(index, 1.0);
						this.m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
					this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
				}
				//if the token is not in the vocabulary, nothing to do.
			}
			doc.createSpVct(spVct);
			doc.L2Normalization(doc.getSparse());
			m_corpus.addDoc(doc);
			this.m_corpus.sizeAddOne();
			this.m_classMemberNo[doc.getYLabel()]++;
		}catch(Exception e) {e.printStackTrace();}
		//System.out.print(".");
	}
	
	//Sort the documents.
	public void setTimeFeatures(int window){
		this.m_window = window;
		ArrayList<_Doc> docs = new ArrayList<_Doc>(this.m_corpus.getCollection());
		//Sort the documents according to time stamps.
		Collections.sort(docs, new Comparator<_Doc>(){
			public int compare(_Doc d1, _Doc d2){
				if(d1.getTimeStamp() == d2.getTimeStamp())
					return 0;
					return d1.getTimeStamp() < d2.getTimeStamp() ? -1 : 1;
			}
		});
		/************************time series analysis***************************/
		for(int i = 0; i < docs.size(); i++){
			_SparseFeature[] tempVct = docs.get(i).getSparse();
			if(this.m_YLabelQueue.size() < m_window){
				this.m_YLabelQueue.add(docs.get(i).getYLabel());
				this.m_SpVctQueue.add(docs.get(i).getSparse());
				this.m_corpus.removeDoc(i);
				this.m_corpus.sizeMinusOne();
				this.m_classMemberNo[docs.get(i).getYLabel()]--;
			}
			else{
				if(this.m_YLabelQueue.size() == m_window && this.m_SpVctQueue.size() == m_window){
					docs.get(i).createSpVctWithTime(this.m_YLabelQueue, this.m_SpVctQueue, this.m_featureNames.size());
					// doc.L2Normalization(doc.getSparse());//Normalize the sparse.
					this.m_YLabelQueue.remove();
					this.m_SpVctQueue.remove();
					this.m_YLabelQueue.add(docs.get(i).getYLabel());
					this.m_SpVctQueue.add(tempVct);
				}
			}
		}
	}
}
