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
import org.jsoup.Jsoup;

import structures.TokenizeResult;

import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import utils.Utils;

public class ParentChildAnalyzer extends jsonAnalyzer {
	public HashMap<String, _ParentDoc> parentHashMap;

	public static int ChildDocFeatureSize = 6;
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

	
	public void loadChildDoc(String fileName) {
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJson(fileName);
		System.out.println("fileName\t" + fileName);
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");
		String title = Utils.getJSONValue(json, "title");

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
	
	public void setFeatureValues(String fValue, int norm){
		super.setFeatureValues(fValue, norm);
		
		ArrayList<_Doc> corpusDocList = new ArrayList<_Doc>();
		corpusDocList = m_corpus.getCollection();
		
		int N = corpusDocList.size(); // total number of documents
		int childDocsNum = 0;
		int parentDocsNum = 0;
		
		int vocabulary_size = m_featureNames.size();
		
		double[] childDF = new double[vocabulary_size]; // total number of unique words
		double[] corpusDF = new double[vocabulary_size];
		double[] parentDF = new double[vocabulary_size];

		double totalWords = 0;
		for(_Doc temp:corpusDocList) {
			if(temp instanceof _ParentDoc){
				double pChildWordNum = 0;
				_SparseFeature[] pfs = temp.getSparse();
				for(_SparseFeature sf : pfs){
					parentDF[sf.getIndex()] ++;	// DF in child documents
					corpusDF[sf.getIndex()] ++;
				}
				parentDocsNum += 1;
				
				for(_ChildDoc cDoc:((_ParentDoc) temp).m_childDocs){
					_SparseFeature[] cfs = cDoc.getSparse();
					for(_SparseFeature sf : cfs){
						childDF[sf.getIndex()] ++;	// DF in child documents
						corpusDF[sf.getIndex()] ++;
					}
					childDocsNum += 1;
					totalWords += temp.getTotalDocLength();
				}
			}
		}
		
		System.out.println("totalWords\t"+totalWords);
		System.out.println("Set feature value for parent child probit model");
		_SparseFeature[] parentFvs;
		for(_Doc tempDoc:corpusDocList) {	
			if(tempDoc instanceof _ParentDoc) {
				parentFvs = tempDoc.getSparse();
				_ParentDoc tempParentDoc = (_ParentDoc)tempDoc;
				tempParentDoc.initFeatureWeight(ChildDocFeatureSize);
				
				for(_ChildDoc tempChildDoc:((_ParentDoc) tempDoc).m_childDocs){
					_SparseFeature[] childFvs = tempChildDoc.getSparse();
					for(_SparseFeature sf: childFvs){
						int wid = sf.getIndex();
						
						double DFCorpus = corpusDF[wid];
						double IDFCorpus = DFCorpus>0 ? Math.log((N+1)/DFCorpus):0;
						
						double[] values = new double[ChildDocFeatureSize];
						
						double DFChild = childDF[wid];
						double IDFChild = DFChild>0 ? Math.log((childDocsNum+1)/DFChild):0;
						
						values[0] = 1;
						values[1] = IDFCorpus;
						
						double TFParent = 0;
						double TFChild = 0;
						
						int wIndex = Utils.indexOf(parentFvs, wid);
						if(wIndex != -1){
							TFParent = parentFvs[wIndex].getValue();	
						}
						
						TFChild = sf.getValue();

						values[2] = TFParent;//TF in parent document
						values[3] = TFChild;//TF in child document					
						values[4] = TFParent/TFChild;//TF ratio
						
						values[5] = IDFCorpus * TFChild;//TF-IDF
						sf.setValues(values);
					}
				}
			
			}	
		}

	}
	
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
	
	
}
