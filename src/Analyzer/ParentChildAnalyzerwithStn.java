package Analyzer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.TokenizeResult;
import structures._ChildDoc;
import structures._ParentDoc;
import structures._Stn;
import structures._stat;
import utils.Utils;

public class ParentChildAnalyzerwithStn extends ParentChildAnalyzer{
	public HashMap<String, _ParentDoc> parentHashMap;

	public ParentChildAnalyzerwithStn(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		parentHashMap = new HashMap<String, _ParentDoc>();
	}

	public void LoadParentDirectory(String folder, String suffix) {
		if(folder==null||folder.isEmpty())
			return;
		
		int current = m_corpus.getSize();
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			if(f.isFile() && f.getName().endsWith(suffix)){
				loadParentDoc(f.getAbsolutePath());
			}else if(f.isDirectory()){
				LoadParentDirectory(folder, suffix);
			}
		}
		System.out.format("loading %d news from %s\n", m_corpus.getSize()-current, folder);
	}
	
	public void LoadChildDirectory(String folder, String suffix) {
		if(folder==null||folder.isEmpty())
			return;
		
		int current = m_corpus.getSize();
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			if(f.isFile() && f.getName().endsWith(suffix)){
				loadChildDoc(f.getAbsolutePath());
			}else if(f.isDirectory()){
				LoadChildDirectory(folder, suffix);
			}
		}
		
		System.out.format("loading %d comments from %s\n", m_corpus.getSize()-current, folder);
	}

	public void loadParentDoc(String fileName) {
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJson(fileName);
		String title = Utils.getJSONValue(json, "title");
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");

		_ParentDoc d = new _ParentDoc(m_corpus.getSize(), name, title, content, 0);
		if (!m_isCVLoaded) {
			AnalyzeDoc(d);
		} else {
			JSONArray sentenceArray;
			try {
				sentenceArray = json.getJSONArray("sentences");
				analyzerParentDoc(d, sentenceArray);

			} catch (JSONException e) {
				e.printStackTrace();
			}
		}

		if (m_corpus.getCollection().contains(d)) {
			parentHashMap.put(name, d);
		}

	}

	public void loadChildDoc(String fileName) {
		if (fileName == null) {
			return;
		}

		JSONObject json = LoadJson(fileName);
		String title = Utils.getJSONValue(json, "title");
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");

		_ChildDoc d = new _ChildDoc(m_corpus.getSize(), name, title, content, 0);
		
		AnalyzeDoc(d);

		if (m_corpus.getCollection().contains(d)) {
			if (parentHashMap.containsKey(parent)) {
				_ParentDoc pDoc = parentHashMap.get(parent);
				d.setParentDoc(pDoc);
				pDoc.addChildDoc(d);
			}
		}
	}
	
	public void analyzerParentDoc(_ParentDoc doc, JSONArray sentenceArray) {
		TokenizeResult result;
		int y = doc.getYLabel();

		String[] sentences = new String[sentenceArray.length()];
		try {
			for (int i = 0; i < sentenceArray.length(); i++) {
				String sentence = Utils.getJSONValue(sentenceArray.getJSONObject(i), "sentence");
				sentences[i] = sentence;
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}

		// Collect the index and counts of features.
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>();
		ArrayList<_Stn> stnList = new ArrayList<_Stn>(); // sparse sentence

		HashMap<Integer, _Stn> stnMap = new HashMap<Integer, _Stn>();

		double stopwordCnt = 0, rawCnt = 0;

		for (String sentence : sentences) {
			result = TokenizerNormalizeStemmer(sentence);
			
			int index = 0;
			double value = 0;
			// Collect the index and counts of features.
			HashMap<Integer, Double> sentence_vector = new HashMap<Integer, Double>();
			for (String token : result.getTokens()) {
				if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
			
					index = m_featureNameIndex.get(token);

					if (sentence_vector.containsKey(index)) {
						value = sentence_vector.get(index) + 1;
						sentence_vector.put(index, value);
					} else {
						sentence_vector.put(index, 1.0);

						m_featureStat.get(token).addOneDF(y);
					}
					m_featureStat.get(token).addOneTTF(y);
				}
			}

			if (sentence_vector.size() > 2) {// avoid empty sentence
				String[] posTags = null;

				_Stn stnObj = new _Stn(Utils.createSpVct(sentence_vector), result.getRawTokens(), posTags, sentence);
				stnList.add(stnObj);
				Utils.mergeVectors(sentence_vector, spVct);

				stopwordCnt += result.getStopwordCnt();
				rawCnt += result.getRawCnt();

				//added by Renqin
				//used to locate which sentence is filtered
				int sentenceID = stnMap.size();
				stnMap.put(sentenceID, stnObj);

			} else {
				//roll back short sentences

				// all short sentences stored in sentenceMap but length to be 0
				int sentenceID = stnMap.size();
				_Stn sentenceObj = null;

				stnMap.put(sentenceID, sentenceObj);

				if (sentence_vector.size() != 0) {
					for (int keyIndex : sentence_vector.keySet()) {
						String token = m_featureNames.get(keyIndex);
						_stat stat = m_featureStat.get(token);
						stat.minusOneDF(y);
						stat.minusNTTF(y, sentence_vector.get(keyIndex));
					}
				}

			}
		} // End For loop for sentence

		// the document should be long enough
		if (spVct.size() >= m_lengthThreshold) {
			doc.createSpVct(spVct);
			doc.setStopwordProportion(stopwordCnt / rawCnt);
			doc.setSentences(stnList);
			doc.m_sentenceMap = stnMap;

			m_corpus.addDoc(doc);
			m_classMemberNo[y]++;

			if (m_releaseContent)
				doc.clearSource();

		} else {
			/**** Roll back here!! ******/
			rollBack(spVct, y);
		}
	}
}
