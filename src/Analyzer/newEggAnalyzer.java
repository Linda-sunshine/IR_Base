/**
 * 
 */
package Analyzer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.NewEggPost;
import structures._Doc;

/**
 * @author hongning
 * For specific format of NewEgg reviews
 */
public class newEggAnalyzer extends jsonAnalyzer {
	//category of NewEgg reviews
	String m_category; 
	
	SimpleDateFormat m_dateFormatter;
	public newEggAnalyzer(String tokenModel, int classNo, String providedCV,
			int Ngram, int threshold, String category) throws InvalidFormatException,
			FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		//11/7/2013 7:01:22 PM
		m_dateFormatter = new SimpleDateFormat("M/d/yyyy h:mm:ss a");// standard date format for this project
		m_category = category;
	}

	public newEggAnalyzer(String tokenModel, int classNo, String providedCV,
			int Ngram, int threshold, String stnModel, String category)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, stnModel);
		
		m_category = category;
	}
	
	//Load all the files in the directory.
	public void LoadNewEggDirectory(String folder, String suffix) throws IOException {
		if (folder==null || folder.isEmpty())
			return;
		
		int current = m_corpus.getSize();
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				LoadNewEggDoc(f.getAbsolutePath());
			} else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		System.out.format("Loading %d reviews from %s\n", m_corpus.getSize()-current, folder);
	}

	//Load a document and analyze it.
	public void LoadNewEggDoc(String filename) {
		JSONObject prods = null;
		String item;
		JSONArray itemIds, reviews;
		
		try {
			JSONObject json = LoadJson(filename);
			prods = json.getJSONObject(m_category);
			itemIds = prods.names();
		} catch (Exception e) {
			System.out.print('X');
			return;
		}	
		
		for(int i=0; i<itemIds.length(); i++) {
			try {
				item = itemIds.getString(i);
				reviews = prods.getJSONArray(item);
				for(int j=0; j<reviews.length(); j++) 
					AnalyzeNewEggPost(new NewEggPost(reviews.getJSONObject(j), item));
			} catch (JSONException e) {
				System.out.print('P');
			} catch (ParseException e) {
				e.printStackTrace();
			}
		}
	}
	
	protected boolean AnalyzeNewEggPost(NewEggPost post) throws ParseException {
		String[] tokens;
		String content;
		StringBuffer buffer = m_releaseContent?null:new StringBuffer(256);
		HashMap<Integer, Double> vPtr;
		ArrayList<HashMap<Integer, Double>> spVcts = new ArrayList<HashMap<Integer, Double>>(); // Collect the index and counts of features.
		int y = post.getLabel()-1, uniWordsInSections = 0;
		
		if ((content=post.getProContent()) != null) {// tokenize pros
			tokens = TokenizerNormalizeStemmer(content);
			vPtr = constructSpVct(tokens, y);
			spVcts.add(vPtr);
			uniWordsInSections += vPtr.size();
			
			if (!m_releaseContent)
				buffer.append(String.format("Pros: %s\n", content));
		} else 
			spVcts.add(null);//no pro section
		
		if ((content=post.getConContent()) != null) {// tokenize cons
			tokens = TokenizerNormalizeStemmer(content);
			vPtr = constructSpVct(tokens, y);
			spVcts.add(vPtr);
			uniWordsInSections += vPtr.size();
			
			if (!m_releaseContent)
				buffer.append(String.format("Cons: %s\n", content));
		} else 
			spVcts.add(null);//no con section
		
		if ((content=post.getComments()) != null) {// tokenize comments
			tokens = TokenizerNormalizeStemmer(content);
			vPtr = constructSpVct(tokens, y);
			spVcts.add(vPtr);
			uniWordsInSections += vPtr.size();
			
			if (!m_releaseContent)
				buffer.append(String.format("Comments: %s\n", content));
		} else
			spVcts.add(null);//no comments
		
		if (uniWordsInSections>=m_lengthThreshold) {
			long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
			_Doc doc = new _Doc(m_corpus.getSize(), post.getID(), (m_releaseContent?null:buffer.toString()), post.getProdId(), y, timeStamp);			
			
			doc.createSpVct(spVcts);
			m_corpus.addDoc(doc);
			m_classMemberNo[y]++;
			return true;
		} else
			return false;
	}
}
