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
import structures._ChildDocProbitModel;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

public class ParentChildAnalyzer extends jsonAnalyzer {
	public HashMap<String, _ParentDoc> parentHashMap;

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

	public void loadChildDoc(String fileName) {
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJson(fileName);
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");

		_ChildDoc d = new _ChildDoc(m_corpus.getSize(), name, "", content, 0);
		
		if (AnalyzeDoc(d)) {//this is a valid child document
			if (parentHashMap.containsKey(parent)) {
				_ParentDoc pDoc = parentHashMap.get(parent);
				d.setParentDoc(pDoc);
				pDoc.addChildDoc(d);
			} else {
				System.err.format("[Warning]Missing parent document %s!\n", parent);
			}			
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
	
	@Override
	public void setFeatureValues(String fValue, int norm){
		super.setFeatureValues(fValue, norm);//it is safe to call this first
		
		ArrayList<_Doc> docs = m_corpus.getCollection(); // Get the collection of all the documents.
		int N = m_isCVStatLoaded ? m_TotalDF : docs.size();
		int childDocsNum = 0;
		
		_stat[] childFeatureStat = new _stat[m_featureNames.size()];
		
		for(_Doc temp:docs) {
			if(temp instanceof _ChildDoc){
				_SparseFeature[] sfs = temp.getSparse();
				for(_SparseFeature sf : sfs)
					childFeatureStat[sf.getIndex()].addOneDF(temp.getYLabel());				
				childDocsNum += 1;
			}
		}
		
		for(_Doc temp:docs) {
			_SparseFeature[] sfs = temp.getSparse();
			double avgIDF = 0.0;
			
			for(_SparseFeature sf: sfs){
				String featureName = m_featureNames.get(sf.getIndex());
				_stat stat = m_featureStat.get(featureName);
				
				double DFCorpus = Utils.sumOfArray(stat.getDF());
				double IDFCorpus = DFCorpus>0 ? Math.log((N+1)/DFCorpus):0;
				avgIDF += IDFCorpus;
				
				if(temp instanceof _ChildDocProbitModel){
					double[] values = new double[6];
					
					_stat childStat = childFeatureStat[sf.getIndex()];
					double DFChild = Utils.sumOfArray(childStat.getDF());
					double IDFChild = DFChild>0 ? Math.log((childDocsNum+1)/DFChild):0;
					
					values[0] = IDFCorpus;
					values[1] = IDFChild;
					values[2] = IDFChild==0 ? 0:IDFCorpus/IDFChild;
					
					double TFParent = 0.0;
					double TFChild = 0.0;
							
					_ParentDoc tempParentDoc = ((_ChildDocProbitModel)temp).m_parentDoc;
					for(_SparseFeature sfParent: tempParentDoc.getSparse()){
						if(sfParent.getIndex() == sf.getIndex()) {
							TFParent = sfParent.getValue();
							break;
						}
					}
					
					TFChild = sf.getValue();
					values[3] = TFParent;
					values[4] = TFChild;
					values[5] = TFParent/TFChild;
					
					sf.setValues(values);
				}
					
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		}
		
	}
	
}
