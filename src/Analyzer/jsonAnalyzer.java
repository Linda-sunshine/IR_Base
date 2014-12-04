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
import java.text.Normalizer;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import opennlp.tools.util.InvalidFormatException;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;


import structures.Post;
import structures._Doc;
import structures._SparseFeature;
import structures._stat;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class jsonAnalyzer extends Analyzer{
	int m_Ngram;
	private int m_window; //The length of the window which means how many labels will be taken into consideration.
	private LinkedList<Integer> m_YLabelQueue;
	private LinkedList<_SparseFeature[]> m_SpVctQueue;
	
	ArrayList<JSONObject> m_threads;
	SimpleDateFormat m_dateFormatter;

	
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, String fs) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		if(providedCV != null)
			this.LoadCV(providedCV);
		if(fs != null){
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}	
		this.m_Ngram = 1;
		m_dateFormatter = new SimpleDateFormat("mmmmmmmmm dd,yyyy");//standard date format for this project
	}	
	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, String fs, int Ngram) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		if (providedCV != null)
			this.LoadCV(providedCV);
		if (fs != null) {
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}
		this.m_Ngram = Ngram;
		m_threads = new ArrayList<JSONObject>();
		m_dateFormatter = new SimpleDateFormat("mmmmmmmmm dd,yyyy");//standard date format for this project
	}

	// Constructor with ngram and time series analysis.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, String fs, int Ngram, boolean timeFlag, int window) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo);
		if (providedCV != null)
			this.LoadCV(providedCV);
		if (fs != null) {
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}
		this.m_Ngram = Ngram;
		this.m_timeFlag = timeFlag;
		this.m_window = window;
		this.m_YLabelQueue = new LinkedList<Integer>();
		this.m_SpVctQueue = new LinkedList<_SparseFeature[]>();
		m_threads = new ArrayList<JSONObject>();
		m_dateFormatter = new SimpleDateFormat("mmmmmmmmm dd,yyyy");//standard date format for this project
	}
	
	//Load the features from a file and store them in the m_featurNames.@added by Lin.
	public boolean LoadCV(String filename) {
		int count = 0;
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				this.m_featureNames.add(line);
			}
			reader.close();
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
			return false;
		}
		// Indicate we can only use the loaded features to construct the feature
		m_isCVLoaded = true;

		// Set the index of the features.
		for (String f : this.m_featureNames) {
			this.m_featureNameIndex.put(f, count);
			this.m_featureIndexName.put(count, f);
			this.m_featureStat.put(f, new _stat(this.m_classNo));
			count++;
		}
		return true; // if loading is successful
	}
	
	public void LoadDirectory(String folder, String suffix) {
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
	public void AnalyzeThreadedDiscussion(JSONObject json) {		
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post post = new Post(jarray.getJSONObject(i));
				checkPostFormat(post);
				AnalyzeDoc(new _Doc(m_corpus.getSize(), post.getContent(), post.getLabel(), post.getDate()));
				System.out.println(post.getDate());
				//this.m_classMemberNo[post.getLabel()-1]++;
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	//check format for each post
	private void checkPostFormat(Post p) {
		if (p.getLabel() <= 0 || p.getLabel() > 5)
			System.err.println("[Error]Missing Lable or wrong label!!");
		else if (p.getContent() == null)
			System.err.format("[Error]Missing content!!\n");
		else if (p.getDate() == null)
			System.err.format("[Error]Missing date!!\n");
		else {
			// to check if the date format is correct
			try {
				m_dateFormatter.parse(p.getDate());
				System.out.println(m_dateFormatter);
			} catch (ParseException e) {
				System.err.format("[Error]Wrong date format!", p.getDate());
			}
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
			/************************time series analysis***************************/
			//If the timeflag is not set.
			if(!this.m_timeFlag){
				doc.createSpVct(spVct);
				doc.L2Normalization(doc.getSparse());
				m_corpus.addDoc(doc);
				this.m_corpus.sizeAddOne();
				this.m_classMemberNo[(doc.getYLabel()-1)]++;
			}
			//If the timeflag is set.
			else { 
				if(this.m_YLabelQueue.size() < m_window){
					this.m_YLabelQueue.add(doc.getYLabel());
					this.m_SpVctQueue.add(doc.createSpVct(spVct));
				}
				else{
					this.m_YLabelQueue.add(doc.getYLabel());
					this.m_SpVctQueue.add(doc.createSpVct(spVct));
					this.m_SpVctQueue.remove();
					this.m_YLabelQueue.remove();
				}
				if(this.m_YLabelQueue.size() == m_window && this.m_SpVctQueue.size() == m_window){
					doc.createSpVctWithTime(this.m_YLabelQueue, this.m_SpVctQueue, this.m_featureNames.size());
					// doc.L2Normalization(doc.getSparse());//Normalize the sparse.
					m_corpus.addDoc(doc);
					this.m_corpus.sizeAddOne();
					this.m_classMemberNo[doc.getYLabel()-1]++;
				}
			}
		}catch(Exception e) {e.printStackTrace();}
	}
	
	//Tokenizer.
	public String[] Tokenizer(String source) {
		String[] tokens = m_tokenizer.tokenize(source);
		return tokens;
	}
	// Normalize.
	public String Normalize(String token) {
		token = Normalizer.normalize(token, Normalizer.Form.NFKC);
		token = token.replaceAll("\\W+", "");
		token = token.toLowerCase();
		return token;
	}
	// Snowball Stemmer.
	public String SnowballStemming(String token) {
		m_stemmer.setCurrent(token);
		if (m_stemmer.stem())
			return m_stemmer.getCurrent();
		else
			return token;
	}
	// Given a long string, tokenize it, normalie it and stem it, return back the string array.
	public String[] TokenizerNormalizeStemmer(String source) {
		String[] tokens = Tokenizer(source); // Original tokens.
		// Normalize them and stem them.
		for (int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]));

		int tokenLength = tokens.length, N = this.m_Ngram, NgramNo = 0;
		ArrayList<String> Ngrams = new ArrayList<String>();

		// Collect all the grams, Ngrams, N-1grams...
		while (N > 0) {
			NgramNo = tokenLength - N + 1;
			for (int i = 0; i < NgramNo; i++) {
				StringBuffer Ngram = new StringBuffer(128);
				for (int j = 0; j < N; j++) {
					if (j == 0)
						Ngram.append(tokens[i + j]);
					else
						Ngram.append("-" + tokens[i + j]);
				}
				Ngrams.add(Ngram.toString());
			}
			N--;
		}
		return Ngrams.toArray(new String[Ngrams.size()]);
	}
	
	//Add one more token to the current vocabulary.
	private void expandVocabulary(String token) {
		m_featureNames.add(token); // Add the new feature.
		m_featureNameIndex.put(token, (m_featureNames.size() - 1)); // set the index of the new feature.
		m_featureIndexName.put((m_featureNames.size() - 1), token);
	}
	
	//With a new feature added into the vocabulary, add the stat into stat arraylist.
	public void updateFeatureStat(String token){
		this.m_featureStat.put(token, new _stat(this.m_classNo));
	}
}
