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
import structures._ChildDoc2;
import structures._ParentDoc2;
import structures._Stn;
import structures._stat;
import utils.Utils;

public class ParentChildAnalyzer2 extends ParentChildAnalyzer {
	public HashMap<String, _ParentDoc2> parentHashMap;

	public ParentChildAnalyzer2(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		// TODO Auto-generated constructor stub
		parentHashMap = new HashMap<String, _ParentDoc2>();
	}


	public void loadParentDoc(String fileName) {
		if (fileName == null) {
			return;
		}

		JSONObject json = LoadJson(fileName);
		String title = Utils.getJSONValue(json, "title");
		String content = Utils.getJSONValue(json, "content");

		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");
		String child = Utils.getJSONValue(json, "child");

		_ParentDoc2 d = null;
		int yLabel = m_classNo - 1;

		// initial parent doc
		d = new _ParentDoc2(m_corpus.getSize(), name, title, content, yLabel);

		if (!m_isCVLoaded) {
			AnalyzeDoc(d);

		} else {
			JSONArray sentenceArray;
			try {
				sentenceArray = json.getJSONArray("sentences");
				analyzerParentDoc(d, sentenceArray);

			} catch (JSONException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
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
		String child = Utils.getJSONValue(json, "child");

		_ChildDoc2 d = null;
		int yLabel = m_classNo - 1;

		d = new _ChildDoc2(m_corpus.getSize(), name, title, content, yLabel);

		if (!m_isCVLoaded) {
			AnalyzeDoc(d);
		} else {

			analyzeChildDoc(d);

			if (m_corpus.getCollection().contains(d)) {
				if (parentHashMap.containsKey(parent)) {
					_ParentDoc2 pDoc = parentHashMap.get(parent);
					d.setParentDoc2(pDoc);
					pDoc.addChildDoc(d);
				}
			}

		}

	}

	public void analyzerParentDoc(_ParentDoc2 doc, JSONArray sentenceArray) {
		TokenizeResult result;
		int y = doc.getYLabel();

		String[] sentences = new String[sentenceArray.length()];
		try {

			for (int i = 0; i < sentenceArray.length(); i++) {
				String sentence = Utils.getJSONValue(
						sentenceArray.getJSONObject(i), "sentence");
				sentences[i] = sentence;
			}
		} catch (JSONException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		// Collect the index and counts of features.
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>();
		ArrayList<_Stn> stnList = new ArrayList<_Stn>(); // sparse sentence
															// feature vectors
		double stopwordCnt = 0, rawCnt = 0;
		int docSize = 0;

		HashMap<Integer, _Stn> stnMap = new HashMap<Integer, _Stn>();
		
		for (String sentence : sentences) {
			result = TokenizerNormalizeStemmer(sentence);
			ArrayList<Integer> wordPositionInDoc = new ArrayList<Integer>();
			ArrayList<String> rawTokens = new ArrayList<String>();
			ArrayList<Integer> words = new ArrayList<Integer>();
			int stnLen = 0;

			int index = 0;
			double value = 0;
			// Collect the index and counts of features.
			HashMap<Integer, Double> sentence_vector = new HashMap<Integer, Double>();
			for (String token : result.getTokens()) {
				if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
					wordPositionInDoc.add(docSize);
					docSize += 1;
					stnLen += 1;
					rawTokens.add(token);

					index = m_featureNameIndex.get(token);

					words.add(index);

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

				int sentenceID = stnMap.size();
				_Stn sentenceObj = new _Stn(stnLen);

				sentenceObj.setWordInStn(wordPositionInDoc, rawTokens, words);
				stnMap.put(sentenceID, sentenceObj);
				
				Utils.mergeVectors(sentence_vector, spVct);

				stopwordCnt += result.getStopwordCnt();
				rawCnt += result.getRawCnt();

			} else {
				//roll back short sentences
				docSize -= stnLen;

				// all short sentences stored in sentenceMap but length to be 0
				int sentenceID = stnMap.size();
				_Stn sentenceObj = new _Stn(0);

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



	/***
	 * add an attribute to _ChildDoc2, m_index, used to record the position of
	 * each word
	 ****/
	public void analyzeChildDoc(_ChildDoc2 doc) {
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource());// Three-step
																			// analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();

		int index = 0;
		double value = 0;
		ArrayList<Integer> indexList = new ArrayList<Integer>();
		// Collect the index and counts of features.
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>();

		for (String token : tokens) {
			// tokens could come from a sentence or a document
			// CV is not loaded, take all the tokens as features.
			if (m_featureNameIndex.containsKey(token)) {// CV is loaded.

				/**** make a record of positions of words ****/
				index = m_featureNameIndex.get(token);
				indexList.add(index);
				

				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);

					m_featureStat.get(token).addOneDF(y);
				}
				m_featureStat.get(token).addOneTTF(y);
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		
		if (spVct.size() >= m_lengthThreshold) {// temporary code for debugging
												// purpose
			doc.createSpVct(spVct);
			doc.setStopwordProportion(result.getStopwordProportion());
			doc.setIndex(indexList);

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
