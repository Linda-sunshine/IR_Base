package Analyzer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures._ChildDoc;
import structures._ParentDoc;
import structures._Stn;
import structures._stat;
import utils.Utils;

public class ParentChildAnalyzer extends jsonAnalyzer {
	public HashMap<String, _ParentDoc> parentHashMap;

	public ParentChildAnalyzer(String tokenModel, int classNo,
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
	
		AnalyzeDoc(d);

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
	

}
