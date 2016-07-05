package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;

import org.jsoup.Jsoup;

import structures.TokenizeResult;
import structures._APPQuery;
import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4APP;
import structures._SparseFeature;
import structures._Word;
import structures._stat;
import utils.Utils;

public class ParentChildAnalyzer extends jsonAnalyzer {
	public HashMap<String, _ParentDoc> parentHashMap;
	public ArrayList<_APPQuery> m_Queries;

	public ParentChildAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		//added by Renqin
		//null used to initialize stnModel and posModel
		super(tokenModel, classNo, providedCV, Ngram, threshold, null, null);
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
//				loadAPPParentDoc(f.getAbsolutePath());
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
				// LoadDoc(f.getAbsolutePath());
				loadChildDoc(f.getAbsolutePath());
			}else if(f.isDirectory()){
				LoadChildDirectory(folder, suffix);
			}
		}
		System.out.format("loading %d comments from %s\n", m_corpus.getSize()
				- current, folder);

		filterParentAndChildDoc();
	}

	public void loadParentDoc(String fileName) {
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJson(fileName);
		String title = Utils.getJSONValue(json, "title");
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String[] sentences = null;

		_ParentDoc d = new _ParentDoc(m_corpus.getSize(), name, title, content, 0);

		try {
			JSONArray sentenceArray = json.getJSONArray("sentences");
				
			sentences = new String[sentenceArray.length()];
			//shall we add title into this sentence array
			for (int i = 0; i < sentenceArray.length(); i++)
				sentences[i] = Utils.getJSONValue(sentenceArray.getJSONObject(i), "sentence");
			
			if (AnalyzeDocByStn(d, sentences))
				parentHashMap.put(name, d);

		} catch (JSONException e) {
			e.printStackTrace();
		}
	}

	public void loadAPPParentDoc(String fileName){
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJson(fileName);
		String title = Utils.getJSONValue(json, "title");
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");

		content = Jsoup.parse(content).text();
		_ParentDoc4APP d = new _ParentDoc4APP(m_corpus.getSize(), name, title, content, 0);
		
		if (AnalyzeDoc(d)){
			parentHashMap.put(name, d);
//			System.out.println("parent name\t"+name);
		}else{
			System.out.println("parent name\t"+name+"\t remove");
		}
	}
	
	public void loadChildDoc(String fileName) {
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJson(fileName);
		System.out.println("fileName\t" + fileName);
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");
		String title = Utils.getJSONValue(json, "title");
		
		content = Jsoup.parse(content).text();

//		_ChildDoc4APP d = new _ChildDoc4APP(m_corpus.getSize(), name, title,
//				content, 0);
//		

		_ChildDoc4BaseWithPhi d = new _ChildDoc4BaseWithPhi(m_corpus.getSize(),
				name, "", content, 0);
//		_ChildDoc4BaseWithPhi_Hard d = new _ChildDoc4BaseWithPhi_Hard(m_corpus.getSize(), name, "", content, 0) ;
		// _ChildDoc4ChildPhi d = new _ChildDoc4ChildPhi(m_corpus.getSize(),
		// name,
		// "", content, 0);
//		_ChildDoc4TwoPhi d = new _ChildDoc4TwoPhi(m_corpus.getSize(), name, "", content, 0);
//		_ChildDoc4ThreePhi d = new _ChildDoc4ThreePhi(m_corpus.getSize(), name,
//				"", content, 0);
//		_ChildDoc4OneTopicProportion d = new _ChildDoc4OneTopicProportion(m_corpus.getSize(), name, "", content, 0);
//		 _ChildDoc d = new _ChildDoc(m_corpus.getSize(), name, "", content,
//		 0);

//		_ChildDoc4ProbitModel d = new _ChildDoc4ProbitModel(m_corpus.getSize(), name, "", content, 0);
//		_ChildDoc4LogisticRegression d = new _ChildDoc4LogisticRegression(m_corpus.getSize(), name, "", content, 0);
	
		if(parentHashMap.containsKey(parent)){
			if (AnalyzeDoc(d)) {//this is a valid child document
//			if (parentHashMap.containsKey(parent)) {
				_ParentDoc pDoc = parentHashMap.get(parent);
				d.setParentDoc(pDoc);
				pDoc.addChildDoc(d);
			} else {
//				System.err.format("filtering comments %s!\n", parent);
			}			
		}else {
//			System.err.format("[Warning]Missing parent document %s!\n", parent);
		}	
	}
	
	public void LoadDoc(String fileName){
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJson(fileName);
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");

		_Doc d = new _Doc(m_corpus.getSize(), content, 0);
		d.setName(name);
		AnalyzeDoc(d);		
	}
	
	public void loadQuery(String queryFile){
		try{
			m_Queries = new ArrayList<_APPQuery>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(queryFile), "UTF-8"));
			
			String line;
			while((line=reader.readLine())!=null){
				String[] lineUnit = line.split("\t");
				String querySource = lineUnit[1];
				int queryID = Integer.parseInt(lineUnit[0]);
				
				_APPQuery appQuery = new _APPQuery();
				if(AnalyzeQuery(appQuery, querySource)){
					m_Queries.add(appQuery);
					appQuery.setQueryID(queryID);
					System.out.println("query\t"+querySource+"\t accepted");
				}else{
					System.out.println("query\t"+querySource+"\t removed");
				}
			}
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	protected boolean AnalyzeQuery(_APPQuery appQuery, String source){
		TokenizeResult result = TokenizerNormalizeStemmer(source);
		String[] tokens = result.getTokens();
		int wid = 0;
		
		ArrayList<_Word> wordList = new ArrayList<_Word>();
		
		for(String token:tokens){
			if(!m_featureNameIndex.containsKey(token)){
				continue;
			}
			
			wid = m_featureNameIndex.get(token);
			_Word word = new _Word(wid);
			
			wordList.add(word);
		}
		
		if(wordList.isEmpty())
			return false;
		else{
			appQuery.initWords(wordList);
			return true;
		}
	}
	
	
	
//	public void setFeatureValues(String fValue, int norm){
//		super.setFeatureValues(fValue, norm);
//		
//		ArrayList<_Doc> docs = m_corpus.getCollection(); // Get the collection of all the documents.
//		int N = m_isCVStatLoaded ? m_TotalDF : docs.size();
//		
//		double k1 = 1.5; // [1.2, 2]
//		double b = 0.75; // (0, 1000]
//		double navg = 0;
//		for (int k = 0; k < N; k++)
//			navg += docs.get(k).getTotalDocLength();
//		navg /= N;
//
//		for (int i = 0; i < docs.size(); i++) {
//			_Doc temp = docs.get(i);
//			_SparseFeature[] sfs = temp.getSparse();
//			double n = temp.getTotalDocLength() / navg, avgIDF = 0;
//			for (_SparseFeature sf : sfs) {
//				String featureName = m_featureNames.get(sf.getIndex());
//				_stat stat = m_featureStat.get(featureName);
//				double TF = sf.getValue();
//				double DF = Utils.sumOfArray(stat.getDF());
//				double IDF = Math.log((N - DF + 0.5) / (DF + 0.5));
//				double BM25 = IDF * TF * (k1 + 1) / (k1 * (1 - b + b * n) + TF);
//				double[] values = new double[ChildDocFeatureSize];
//				values[0] = BM25;
//				sf.setValues(values);
//				avgIDF += IDF;
//			}
//		}
//	}
	
	public void filterParentAndChildDoc(){
		System.out.println("before filtering\t"+m_corpus.getSize());
		int corpusSize = m_corpus.getCollection().size();
		ArrayList<Integer> removeIndexList = new ArrayList<Integer>();
		
		int numberOfComments = 0;
		
		for (int i = corpusSize - 1; i > -1; i--) {
			// for (int i = 0; i < corpusSize; i++) {
			_Doc d = m_corpus.getCollection().get(i);
			if(d instanceof _ParentDoc){
				_ParentDoc pDoc = (_ParentDoc)d;
				if(pDoc.m_childDocs.size()>8){
					numberOfComments += 1;
				}
				if(pDoc.m_childDocs.size()==0)
					removeIndexList.add(i);
			}
		}
		
		System.out.println("number of comments > 8 \t"+numberOfComments);
		
		for (int i : removeIndexList) {
			_Doc d = m_corpus.getCollection().get(i);
			System.out
					.println("removing parent without child \t" + d.getName());
			m_corpus.getCollection().remove(i);

		}

		System.out.println("after filtering\t"+m_corpus.getSize());

	}
	
	public void searchAPP(String query){
		for(_Doc d:m_corpus.getCollection()){
			if(d.getTitle().equals(query)){
				System.out.println(query+"\t"+d.getName());
			}
		}
	}
}
