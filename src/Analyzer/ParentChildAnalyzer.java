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
import structures._ChildDoc;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4DCM;
import utils.Utils;

/**
 * 
 * @author Renqin Cai
 * Analyzer to load article comment pairs
 */
public class ParentChildAnalyzer extends DocAnalyzer {
	public HashMap<String, _ParentDoc> parentHashMap;

	public static int ChildDocFeatureSize = 6;
	public ParentChildAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException {
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
				loadChildDoc(f.getAbsolutePath());
			}else if(f.isDirectory()){
				LoadChildDirectory(folder, suffix);
			}
		}
		System.out.format("loading %d comments from %s\n", m_corpus.getSize() - current, folder);

		filterParentAndChildDoc();
	}

	public void loadParentDoc(String fileName) {
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJSON(fileName);
		String title = Utils.getJSONValue(json, "title");
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String[] sentences = null;

		// _ParentDoc d = new _ParentDoc(m_corpus.getSize(), name, title,
		// content, 0);

		_ParentDoc d = new _ParentDoc4DCM(m_corpus.getSize(), name, title,
				content, 0);

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

	
	public void loadChildDoc(String fileName) {
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJSON(fileName);
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");
		String title = Utils.getJSONValue(json, "title");

//		

//		_ChildDoc4BaseWithPhi d = new _ChildDoc4BaseWithPhi(m_corpus.getSize(),
//				name, "", content, 0);
//		_ChildDoc4BaseWithPhi_Hard d = new _ChildDoc4BaseWithPhi_Hard(m_corpus.getSize(), name, "", content, 0) ;
		// _ChildDoc4ChildPhi d = new _ChildDoc4ChildPhi(m_corpus.getSize(),
		// name,
		// "", content, 0);
//		_ChildDoc4TwoPhi d = new _ChildDoc4TwoPhi(m_corpus.getSize(), name, "", content, 0);
//		_ChildDoc4ThreePhi d = new _ChildDoc4ThreePhi(m_corpus.getSize(), name,
//				"", content, 0);
//		_ChildDoc4OneTopicProportion d = new _ChildDoc4OneTopicProportion(m_corpus.getSize(), name, "", content, 0);
		 _ChildDoc d = new _ChildDoc(m_corpus.getSize(), name, "", content,
		 0);

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

		JSONObject json = LoadJSON(fileName);
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");

		_Doc d = new _Doc(m_corpus.getSize(), content, 0);
		d.setName(name);
		AnalyzeDoc(d);		
	}
	
	
	public void filterParentAndChildDoc(){
		System.out.println("Before filtering\t"+m_corpus.getSize());
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

}
