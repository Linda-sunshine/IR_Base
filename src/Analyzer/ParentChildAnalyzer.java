package Analyzer;

import java.io.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import net.didion.jwnl.data.Exc;
import opennlp.tools.util.InvalidFormatException;
import structures.*;
import utils.Utils;

/**
 * 
 * @author Renqin Cai
 * Analyzer to load article comment pairs
 */
public class ParentChildAnalyzer extends DocAnalyzer {
	public HashMap<String, _ParentDoc> parentHashMap;

	public HashMap<String, Integer> labelIntMap = new HashMap<String, Integer>();
	
	public static int ChildDocFeatureSize = 6;
	public ParentChildAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		parentHashMap = new HashMap<String, _ParentDoc>();
		labelIntMap.put("earn", 0);
		labelIntMap.put("acq", 1);
		labelIntMap.put("crude", 2);
		labelIntMap.put("trade", 3);
		labelIntMap.put("money-fx", 4);
		labelIntMap.put("interest", 5);
		labelIntMap.put("ship", 6);
		labelIntMap.put("sugar", 7);
		labelIntMap.put("coffee", 8);
		labelIntMap.put("gold", 9);

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
		String timeStampStr = Utils.getJSONValue(json, "date");
		String name = Utils.getJSONValue(json, "name");
		String[] sentences = null;

//		if(name.equals("49")){
//			System.out.println("49");
//		}

//
//		_ParentDoc d = new _ParentDoc4DCM(m_corpus.getSize(), name, title, content, 0);
//        _ParentDoc4WordEmbedding d = new _ParentDoc4WordEmbedding(m_corpus.getSize(), name, title, content, 0);

		_ParentDocWithRawToken d = new _ParentDocWithRawToken(m_corpus.getSize(), name, title, content, 0);

//		DateFormat formatter = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");
		try {
//			Date date = formatter.parse(timeStampStr);
//			long timeStamp = date.getTime();
//			long timeStamp = Long.parseLong(timeStampStr);
//			d.setTimeStamp(timeStamp);

			JSONArray sentenceArray = json.getJSONArray("sentences");
				
			sentences = new String[sentenceArray.length()];
			//shall we add title into this sentence array
			for (int i = 0; i < sentenceArray.length(); i++) {
				JSONObject jObject = sentenceArray.getJSONObject(i);
				String sentenceID = (String) jObject.keys().next();
//				sentences[i] = Utils.getJSONValue(sentenceArray.getJSONObject(i), "sentence");
				sentences[i] = Utils.getJSONValue(sentenceArray.getJSONObject(i), sentenceID);
			}
			
			if (AnalyzeDocByStn(d, sentences))
				parentHashMap.put(name, d);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void loadChildDocwithTime(String fileName) {
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJSON(fileName);
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");
		String title = Utils.getJSONValue(json, "title");
		String timeStampStr = Utils.getJSONValue(json, "cdate");
		System.out.print("time\t"+timeStampStr);
		DateFormat formatter = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");
		try{
			Date date = formatter.parse(timeStampStr);
			long timeStamp = date.getTime();
			_ChildDoc d = new _ChildDoc(m_corpus.getSize(), name, "", content, 0);
			d.setTimeStamp(timeStamp);

			d.setName(name);

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
		}catch (Exception e){
			System.out.print(fileName);
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

		_ChildDoc d = new _ChildDoc(m_corpus.getSize(), name, "", content, 0);
//		_ChildDoc4BaseWithPhi d = new _ChildDoc4BaseWithPhi(m_corpus.getSize(), name, "", content, 0);
		d.setName(name);

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
			System.err.format("[Warning]Missing parent document %s!\n", parent);
		}
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

//		_ChildDoc d = new _ChildDoc(m_corpus.getSize(), name, "", content, 0);
		// _ChildDoc4DCMDMMCorrLDA d = new _ChildDoc4DCMDMMCorrLDA(
		// m_corpus.getSize(), name, "", content, 0);


//		_ChildDoc4ProbitModel d = new _ChildDoc4ProbitModel(m_corpus.getSize(), name, "", content, 0);
//		_ChildDoc4LogisticRegression d = new _ChildDoc4LogisticRegression(m_corpus.getSize(), name, "", content, 0);
	
//		if(parentHashMap.containsKey(parent)){
//			if (AnalyzeDoc(d)) {//this is a valid child document
////			if (parentHashMap.containsKey(parent)) {
//				_ParentDoc pDoc = parentHashMap.get(parent);
//				d.setParentDoc(pDoc);
//				pDoc.addChildDoc(d);
//			} else {
////				System.err.format("filtering comments %s!\n", parent);
//			}
//		}else {
////			System.err.format("[Warning]Missing parent document %s!\n", parent);
//		}
	}
	
	public void LoadDoc(String fileName){
		boolean DCMLDAFlag = false;
		if(DCMLDAFlag){
			LoadDoc4DCMLDA(fileName);
			return;
		}
		
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJSON(fileName);
//		if(json.has("name")){
//			System.out.println("exist name");
//		}
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");
//		String timeStampStr = Utils.getJSONValue(json, "time");
		String label = Utils.getJSONValue(json, "label");
		
		int yLabel = 0;
		if(!label.equals("NULL")){
//			System.out.println("label\t"+label);
//			yLabel = Integer.parseInt(label);
//			yLabel = labelIntMap.get(label);
		}

		_Doc d = new _Doc(m_corpus.getSize(), content, yLabel);

		d.setName(name);
		AnalyzeDoc(d);

//		DateFormat formatter = new SimpleDateFormat("yyyy-mm-dd");
//        try{
//            Date date = formatter.parse(timeStampStr);
//            long timeStamp = date.getTime();
//            _Doc d = new _Doc(m_corpus.getSize(), content, yLabel);
//            d.setTimeStamp(timeStamp);
//
//            d.setName(name);
//            AnalyzeDoc(d);
//        }catch (Exception e){
//            System.out.print(fileName);
//            e.printStackTrace();
//        }

//		_DynamicDoc d = new _DynamicDoc(m_corpus.getSize(), content, yLabel, timeStamp);

	}
	
	public void LoadDoc4DCMLDA(String fileName){
		if (fileName == null || fileName.isEmpty())
			return;

		JSONObject json = LoadJSON(fileName);
		String content = Utils.getJSONValue(json, "content");
		String name = Utils.getJSONValue(json, "name");
		String parent = Utils.getJSONValue(json, "parent");

//		 _Doc4DCMLDA d = new _Doc4DCMLDA(m_corpus.getSize(), name, "", content, 0);
		String label = Utils.getJSONValue(json, "label");
		
		int yLabel = 0;
		if(!label.equals("NULL")){
//			System.out.println("label\t"+label);
//			yLabel = Integer.parseInt(label);
//			yLabel = labelIntMap.get(label);
		}
		_Doc4SparseDCMLDA d = new _Doc4SparseDCMLDA(m_corpus.getSize(), name, "", content, yLabel);
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

		int totalParentNum = 0;
		for(_Doc d:m_corpus.getCollection()){
			if (d instanceof _ParentDoc){
				totalParentNum += 1;
			}
		}
		System.out.println("after filtering\t"+m_corpus.getSize()+"\t article Num:"+totalParentNum+"\t comment num:"+(m_corpus.getSize()-totalParentNum));
	}

	public void analyzeBurstiness(String filePrefix){
		HashMap<Double, Double> burstinessMap = new HashMap<Double, Double>();
		
		String fileName = filePrefix+"/burstiness.txt";
		int vocalSize = m_corpus.getFeatureSize();
		System.out.println("vocal size\t"+vocalSize);
		
		int corpusSize = 0;
		double threadLen = 0;
		double threadNum = 0;

		double totalFeatureTimes = 0;

		for(_Doc d:m_corpus.getCollection()){
			if(d instanceof _ParentDoc4DCM){
				threadNum ++;
				corpusSize ++;
				HashMap<Integer, Double> wordFrequencyMap = new HashMap<Integer, Double>();
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				threadLen += pDoc.getTotalDocLength();

				_SparseFeature[] pDocFS = pDoc.getSparse();
				for(_SparseFeature sf:pDocFS){
					int wid = sf.getIndex();
					double featureTimes = sf.getValue();
					if(!wordFrequencyMap.containsKey(wid))
						wordFrequencyMap.put(wid, 0.0);
//					else{
//						double oldFeatureTimes = wordFrequencyMap.get(wid);
//						oldFeatureTimes += featureTimes;
//						wordFrequencyMap.put(wid, oldFeatureTimes);
//					}

				}

				for(_ChildDoc cDoc:pDoc.m_childDocs){
//					System.out.println("cDoc\t"+cDoc.getName());
					threadLen += cDoc.getTotalDocLength();
					for(_Word w:cDoc.getWords()){
						int wid = w.getIndex();
//						System.out.print("wid\t"+wid+"\t");
						if(!wordFrequencyMap.containsKey(wid)){
							continue;
//							wordFrequencyMap.put(wid, 1.0);
						}else{
							double oldFeaturetimes = wordFrequencyMap.get(wid);
							oldFeaturetimes += 1;
							wordFrequencyMap.put(wid, oldFeaturetimes);
						}
					}
					System.out.print("\n");

//					_SparseFeature[] sfs = cDoc.getSparse();
//					for(_SparseFeature sf:sfs){
//						int wid = sf.getIndex();
//						double featureTimes = sf.getValue();
//						if(!wordFrequencyMap.containsKey(wid))
//							continue;
////							wordFrequencyMap.put(wid, featureTimes);
//						else{
//							double oldFeatureTimes = wordFrequencyMap.get(wid);
//							oldFeatureTimes += featureTimes;
//							wordFrequencyMap.put(wid, oldFeatureTimes);
//						}
//
//					}
				}

				totalFeatureTimes += wordFrequencyMap.size();
//				double zeroWordNum = 0;
//				for(int wid: wordFrequencyMap.keySet()){
//					if(wordFrequencyMap.get(wid)==0.0) {
////						System.out.println(wordFrequencyMap.get(wid) + "\t" + wid);
//						zeroWordNum += 1;
//					}
//				}

//				System.out.println("wordFrequencyMap.size()\t"+wordFrequencyMap.size()+"zeroWordNum\t"+zeroWordNum);

				for(int wid:wordFrequencyMap.keySet()){
					double featureTimes = wordFrequencyMap.get(wid);
					if(!burstinessMap.containsKey(featureTimes))
						burstinessMap.put(featureTimes, 1.0);
					else{
						double value = burstinessMap.get(featureTimes);
						burstinessMap.put(featureTimes, value+1);
					}
						
				}
				
//				if(!burstinessMap.containsKey((double)0)){
//					burstinessMap.put((double)0, zeroWordNum);
//				}else{
//					zeroWordNum += burstinessMap.get((double)0);
//					burstinessMap.put((double)0, zeroWordNum);
//				}
		
			}
			
		}

		threadLen = threadLen/threadNum;
		System.out.println("threadLen\t"+threadLen+"\tthreadNum\t"+threadNum);
		
//		double totalFeatureTimes = vocalSize*corpusSize;
		for(double featureTimes:burstinessMap.keySet()){
			double featureTimesProb = burstinessMap.get(featureTimes)/totalFeatureTimes;
			System.out.println(totalFeatureTimes+"\t"+burstinessMap.get(featureTimes));
			burstinessMap.put(featureTimes, featureTimesProb);
		}
		
		try{
			PrintWriter pw = new PrintWriter(new File(fileName));
			for(double featureTiems:burstinessMap.keySet()){
				pw.println(featureTiems+":"+burstinessMap.get(featureTiems));
			}
			pw.flush();
			pw.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		
	}

	public void generateFakeCorpus(String filePrefix){
		
		HashMap<Integer, Double> t_wordSstat = new HashMap<Integer, Double>();
		double t_allWordFrequency = 0;
		for(_Doc d:m_corpus.getCollection()){
			if(d instanceof _ChildDoc){
				_SparseFeature[] fv = d.getSparse();
				
				for(int i=0; i<fv.length; i++){
					int wid = fv[i].getIndex();
					double val = fv[i].getValue();
					
					t_allWordFrequency += val;
					if(t_wordSstat.containsKey(wid)){
						double oldVal = t_wordSstat.get(wid);
						t_wordSstat.put(wid, oldVal+val);
					}else{
						t_wordSstat.put(wid, val);
					}
				}
			}
		}
		
		for(int wid:t_wordSstat.keySet()){
			double val = t_wordSstat.get(wid);
			double prob = val/t_allWordFrequency;
			t_wordSstat.put(wid, prob);
		}
		
//		languageModelBaseLine lm = new languageModelBaseLine(m_corpus, 0);
//		lm.generateReferenceModel();
		int docIndex = 0;
		
		File fakeCorpusFolder = new File(filePrefix+"fakeCorpus");
		if(!fakeCorpusFolder.exists()){
			System.out.println("creating directory\t"+fakeCorpusFolder);
			fakeCorpusFolder.mkdir();
		}	
		
		ArrayList<Integer> widList = new ArrayList<Integer>(t_wordSstat.keySet());
		
		for(_Doc d:m_corpus.getCollection()){
			if(d instanceof _ParentDoc4DCM){
				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				int docLength = 0;
				generateParentDocFeature(pDoc, fakeCorpusFolder);
//				docLength += pDoc.getTotalDocLength();
				for(_ChildDoc cDoc:pDoc.m_childDocs) {
					docLength = cDoc.getTotalDocLength();
					String docName = cDoc.getName();
					generateFakeDoc(pDoc, fakeCorpusFolder, docLength, t_wordSstat, widList, docName);
				}
			}
		}
	}

	public void generateParentDocFeature(_ParentDoc pDoc, File folder){
		String fakeDocName = pDoc.getName()+".txt";

		try{

			PrintWriter pw = new PrintWriter(new File(folder, fakeDocName));

			for(_SparseFeature sf:pDoc.getSparse()){
				int wid = sf.getIndex();
				double val = sf.getValue();
				for(int i=0; i<val; i++){
					pw.print(wid);
					pw.print("\t");
				}
			}

			pw.flush();
			pw.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void generateFakeDoc(_ParentDoc pDoc, File folder, int docLength, HashMap<Integer, Double>wordSstat, ArrayList<Integer>widList, String docName){
		String fakeDocName = docName+".txt";
		
		try{
			
			PrintWriter pw = new PrintWriter(new File(folder, fakeDocName));
			
//			for(_SparseFeature sf:pDoc.getSparse()){
//				int wid = sf.getIndex();
//				double val = sf.getValue();
//				for(int i=0; i<val; i++){
//					pw.print(wid);
//					pw.print("\t");
//				}
//			}
			
			for(int i=0; i<docLength; i++){
				int wid = generateFakeWord(wordSstat, widList);
//				System.out.println(wid+"wordId");
				pw.print(wid);
				pw.print("\t");
			}
			
			pw.flush();
			pw.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public int generateFakeWord(HashMap<Integer, Double>wordSstat, ArrayList<Integer>widList){
		int wid = 0;
		
		Random t_rand = new Random();
		double prob = t_rand.nextDouble();
		for(int t_wid:widList){
			wid = t_wid;
			double wordProb = wordSstat.get(t_wid);
			prob -= wordProb;
			if(prob<=0){
				break;
			}
		}
		
		return wid;
	}

	public void randomizeComment(String filePrefix, int number_of_topics){
		File fakeCorpusFolder = new File(filePrefix+"fakeCorpus");
		if(!fakeCorpusFolder.exists()){
			System.out.println("creating directory\t"+fakeCorpusFolder);
			fakeCorpusFolder.mkdir();
		}

		ArrayList<_ChildDoc> childList = new ArrayList<_ChildDoc>();
		for(_Doc d:m_corpus.getCollection()){
			if (d instanceof _ParentDoc) {
				d.setTopics4Gibbs(number_of_topics, 0);
			} else if (d instanceof _ChildDoc) {
				_ChildDoc cDoc = (_ChildDoc)d;
				((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, 0);
				childList.add(cDoc);
			}
		}
		System.out.println("child docs num\t"+childList.size());

		int j=0;
		Random m_rand = new Random();
		_Word tempWord;
		for(int i=1; i<childList.size(); i++){
			_ChildDoc d = childList.get(i);
//			System.out.println("cDoc\t"+d.getName());
			for(int wIndex=0; wIndex<d.getTotalDocLength(); wIndex++) {
				_Word w = d.getWordByIndex(wIndex);
				j = m_rand.nextInt(i);
				_Doc tempDoc = childList.get(j);

//				System.out.print("wid\t"+w.getIndex());

				int exchangeDocLen = tempDoc.getTotalDocLength();
				int exchangeWordIndex = m_rand.nextInt(exchangeDocLen);
				_Word exchangeWord = tempDoc.getWordByIndex(exchangeWordIndex);

				tempWord = w;
				w = exchangeWord;
				exchangeWord = tempWord;
				d.setWordByIndex(wIndex, w);
				tempDoc.setWordByIndex(exchangeWordIndex, exchangeWord);
//				System.out.print("--->"+w.getIndex()+"\t"+d.getWordByIndex(wIndex).getIndex());

			}
//			System.out.print("\n");
//			for(_Word w:d.getWords()){
//				System.out.print("wid\t"+w.getIndex());
//			}

		}

//		for(_ChildDoc cDoc:childList){
//			System.out.println("cDoc\t"+cDoc.getName());
//			for(_Word w:cDoc.getWords()) {
//				System.out.print("wid\t"+w.getIndex());
//
//			}
//			System.out.print("\n");
//		}

		System.out.println("randomization");
		analyzeBurstiness(filePrefix);
	}

    public void burstiness4Original(String filePrefix, int number_of_topics){
        for(_Doc d:m_corpus.getCollection()){
            if (d instanceof _ParentDoc) {
                d.setTopics4Gibbs(number_of_topics, 0);
            } else if (d instanceof _ChildDoc) {
                _ChildDoc cDoc = (_ChildDoc)d;
                ((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, 0);
            }
        }
        analyzeBurstiness(filePrefix);
    }

    public void burstiness4Randomized(String filePrefix, int number_of_topics){
        randomizeComment(filePrefix, number_of_topics);
        analyzeBurstiness(filePrefix);
    }

	public void analyzeGeneralizedBurstiness(String filePrefix){

		double SimThreshold = 0.8;//0.08938088820714778,0.3495393872095217
        System.out.print("thresholding\t"+SimThreshold);

		HashMap<Double, Double> burstinessMap = new HashMap<Double, Double>();

		String fileName = filePrefix+"/burstiness.txt";
		int vocalSize = m_corpus.getFeatureSize();
		System.out.println("vocal size\t"+vocalSize);

		int corpusSize = 0;
		double totalFeatureTimes = 0;

		for(_Doc d:m_corpus.getCollection()){
			if(d instanceof  _ParentDoc4DCM){
				corpusSize ++;
				HashMap<Integer, Double> wordFrequencyMap = new HashMap<Integer, Double>();

				_ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
				_SparseFeature[] pDocFS = pDoc.getSparse();

				for(_SparseFeature sf:pDocFS){
					int wid = sf.getIndex();
					String strW = m_corpus.getFeature(wid);
					int wTTF = m_corpus.m_featureStat.get(strW).getTTF()[0];

//					if(wTTF<TTFThreshold){
//						continue;
//					}

					if(!wordFrequencyMap.containsKey(wid)){
						wordFrequencyMap.put(wid, 0.0);
					}

				}

				for(_ChildDoc cDoc:pDoc.m_childDocs){
					for(_Word w:cDoc.getWords()){
						int wid = w.getIndex();
						String strW = m_corpus.getFeature(wid);
						int wTTF = m_corpus.m_featureStat.get(strW).getTTF()[0];

//						if(wTTF>=TTFThreshold){
//							continue;
//						}

						double maxSim = -1;
						int maxWKey = 0;
						for(int wKey:wordFrequencyMap.keySet()){
							double wordSim = getSim4Words(wid, wKey);
							if(wordSim > maxSim) {
								maxWKey = wKey;
                                maxSim = wordSim;
							}
						}

						if(maxSim <= SimThreshold)
						    continue;

						double oldFeatureTimes = wordFrequencyMap.get(maxWKey);
						wordFrequencyMap.put(wid, oldFeatureTimes+1);
					}
				}

				totalFeatureTimes += wordFrequencyMap.size();

				for(int wid:wordFrequencyMap.keySet()){
					double featureTimes = wordFrequencyMap.get(wid);
					if(!burstinessMap.containsKey(featureTimes))
						burstinessMap.put(featureTimes, 1.0);
					else{
						double value = burstinessMap.get(featureTimes);
						burstinessMap.put(featureTimes, value+1);
					}

				}
			}
		}

		for(double featureTimes:burstinessMap.keySet()){
			double featureTimesProb = burstinessMap.get(featureTimes)/totalFeatureTimes;
			burstinessMap.put(featureTimes, featureTimesProb);
		}

		try{
			PrintWriter pw = new PrintWriter(new File(fileName));

			for(double featureTimes: burstinessMap.keySet()){
				pw.println(featureTimes+":"+burstinessMap.get(featureTimes));
			}

			pw.flush();
			pw.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}

	public void loadGloveVec(String gloveFile){
		try{
			String tmpTxt;
			String[] lineContainer;
			double[] featureVecEle;
			int tid = 0;

			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(gloveFile), "UTF-8"));

			while((tmpTxt=br.readLine())!=null){
				tmpTxt = tmpTxt.trim();
				if(tmpTxt.isEmpty())
					continue;

				lineContainer = tmpTxt.split(" ");
				String rawWordStr = lineContainer[0];

				if(rawWordStr.equals("work")) {
					for(int i=0; i<lineContainer.length; i++)
						System.out.print(lineContainer[i]+"\t");
				}
				if(rawWordStr.equals("tricky")) {
					for(int i=0; i<lineContainer.length; i++)
					System.out.print(lineContainer[i]+"\t");
				}
				if(rawWordStr.equals("laboratory")) {
					for(int i=0; i<lineContainer.length; i++)
						System.out.print(lineContainer[i]+"\t");
				}

                TokenizeResult resultToken = TokenizerNormalizeStemmer(rawWordStr);
                if(resultToken.getTokens().length==0)
                    continue;
                String wordStr = resultToken.getTokens()[0];

				if(!m_featureStat.containsKey(wordStr))
					continue;

//                System.out.println(wordStr);
				_stat wordStat = m_featureStat.get(wordStr);

				featureVecEle = new double[lineContainer.length-1];
				for(int i=1; i<lineContainer.length; i++){
					featureVecEle[i-1] = Double.parseDouble(lineContainer[i]);
				}

				wordStat.setM_gloveVec(featureVecEle);
			}

		}catch (Exception e){
			e.printStackTrace();
		}
	}

	public void generateWordSimFile(String gloveFile, String filePrefix){
        String wordSimFile = filePrefix+"wordSim.txt";

        try{
            PrintWriter pw = new PrintWriter(new File(wordSimFile));



        }catch(Exception e){
            e.printStackTrace();
        }
    }

	public double getSim4Words(int pWid, int cWid){
		String pWStr = m_corpus.getFeature(pWid);
		String cWStr = m_corpus.getFeature(cWid);
//		System.out.println("pword\t"+pWStr+"\t cword\t"+cWStr);
        if(m_corpus.m_featureStat.get(pWStr).getM_gloveVec()==null){
            System.out.println(pWStr+"\tno glove vec");
        }

        if(m_corpus.m_featureStat.get(cWStr).getM_gloveVec()==null){
            System.out.println(cWStr+"\tno glove vec");
        }

		double cosSim = Utils.cosine(m_corpus.m_featureStat.get(pWStr).getM_gloveVec(), m_corpus.m_featureStat.get(cWStr).getM_gloveVec());
//		System.out.println(pWStr+"\t"+cWStr+"\tcosSim\t"+cosSim);
		return cosSim;
	}

	public void generalizedBurstiness4Original(String filePrefix, String gloveFile, int number_of_topics){

		ArrayList<_ChildDoc> childList = new ArrayList<_ChildDoc>();
		for(_Doc d:m_corpus.getCollection()){
			if (d instanceof _ParentDoc) {
				d.setTopics4Gibbs(number_of_topics, 0);
			} else if (d instanceof _ChildDoc) {
				_ChildDoc cDoc = (_ChildDoc)d;
				((_ChildDoc) d).setTopics4Gibbs_LDA(number_of_topics, 0);
			}
		}
		loadGloveVec(gloveFile);
		analyzeGeneralizedBurstiness(filePrefix);

	}

	public void generalizedBurstiness4Fake(String filePrefix, String gloveFile, int number_of_topics){
		loadGloveVec(gloveFile);
		randomizeComment(filePrefix, number_of_topics);
		analyzeGeneralizedBurstiness(filePrefix);
	}

	public void simWords4Corpus(String filePrefix, String gloveFile){
        loadGloveVec(gloveFile);
        String wordSimFile = filePrefix+"wordSim_Tech.txt";
        ArrayList<Double> simList = new ArrayList<Double>();

        try{
            PrintWriter pw = new PrintWriter(new File(wordSimFile));

            for(int i=0; i<m_featureNames.size(); i++){
                pw.print(m_featureNames.get(i)+"\t");
            }
            pw.println();

            for(int i=0; i<m_featureNames.size(); i++){
                String wStrRow = m_featureNames.get(i);

                _stat wStatRow = m_featureStat.get(wStrRow);
				if(wStrRow.equals("work")) {
					System.out.println("work\t");
					for(double vecEle: wStatRow.getM_gloveVec())
						System.out.print(vecEle+"\t");
				}

				if(wStrRow.equals("tricki")) {
					System.out.println("tricki\t");
					for(double vecEle: wStatRow.getM_gloveVec())
						System.out.print(vecEle+"\t");
				}

				if(wStrRow.equals("laboratori")) {
					System.out.println("laboratori\t");
					for(double vecEle: wStatRow.getM_gloveVec())
						System.out.print(vecEle+"\t");
				}

//                wStatRow.setSimMap();
                for(int j=0; j<m_featureNames.size(); j++){
                    String wStrCol = m_featureNames.get(j);

                    double cosSim = Utils.cosine(wStatRow.getM_gloveVec(), m_featureStat.get(wStrCol).getM_gloveVec());
    //                wStatRow.m_wordSimMap.put(wStrCol, cosSim);
                    pw.print(cosSim+"\t");
                    simList.add(cosSim);
                }
                pw.println();
            }

            pw.flush();
            pw.close();
        }catch(Exception e){
            e.printStackTrace();
        }


        Collections.sort(simList, Collections.reverseOrder());
        System.out.println("number of sim pairs\t"+simList.size());
        System.out.println("max sim\t"+Collections.max(simList)+"\t"+simList.get(0));
        System.out.println("min sim\t"+Collections.min(simList)+"\t"+simList.get(simList.size()-1));
        int mediumIndex = (int)simList.size()/2;
        int top10Index = (int)(simList.size()*0.1);
        int top20Index = (int)(simList.size()*0.2);
        int top40Index = (int)(simList.size()*0.4);
        System.out.println("medium \t"+simList.get(mediumIndex));
        System.out.println("top 10 sim \t"+simList.get(top10Index));
        System.out.println("top 20 sim \t"+simList.get(top20Index));
        System.out.println("top 40 sim \t"+simList.get(top40Index));

    }

    public void randOutputSim4Comment(String filePrefix, String wordSimFileName){
		int vocabulary_size = m_featureNames.size();

		double[][] m_wordSimMatrix;
		double[] m_wordSimVec;

		m_wordSimMatrix = new double[vocabulary_size][vocabulary_size];
		m_wordSimVec = new double[vocabulary_size];
		loadWordSim4Corpus(wordSimFileName, m_wordSimMatrix, m_wordSimVec, vocabulary_size);


		Random m_rand = new Random();

		ArrayList<_ParentDoc> pDocList = new ArrayList<_ParentDoc>();

		for(_Doc d: m_corpus.getCollection()){
			if(d instanceof _ParentDoc)
				if(d.getName().equals("5"))
					pDocList.add((_ParentDoc) d);
		}

		int randArticleIndex = m_rand.nextInt(pDocList.size());
		_ParentDoc pDoc = pDocList.get(randArticleIndex);

		try {
			for (_ChildDoc cDoc : pDoc.m_childDocs) {
				String commentSimFile = cDoc.getName()+".txt";
				PrintWriter childOut = new PrintWriter(new File(
						filePrefix, commentSimFile));
				for(_SparseFeature sf:cDoc.getSparse()) {
					childOut.print(m_featureNames.get(sf.getIndex())+"\t");
					for(_SparseFeature pSf: pDoc.getSparse()) {
						int pSfVal = (int)pSf.getValue();
						for(int i=0; i< pSfVal; i++)
							childOut.print(m_featureNames.get(pSf.getIndex())+":"+m_wordSimMatrix[sf.getIndex()][pSf.getIndex()]+"\t");
					}
					childOut.println();
				}
				childOut.println();
				childOut.flush();
				childOut.close();
			}
		}catch (Exception e){
			e.printStackTrace();
		}
	}

	public void loadWordSim4Corpus(String wordSimFileName, double[][]m_wordSimMatrix, double[] m_wordSimVec, int vocabulary_size){
		double simThreshold = -1;
		if(wordSimFileName == null||wordSimFileName.isEmpty()){
			return;
		}

//		double[][] m_wordSimMatrix;
//		double[] m_wordSimVec;

		try{
			double maxSim = -2;
			double minSim = 2;

			for(int v=0; v<vocabulary_size; v++) {
				Arrays.fill(m_wordSimMatrix[v], 0);
				Arrays.fill(m_wordSimVec, 0);
			}

			String tmpTxt;
			String[] lineContainer;

			HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
			for(int i=0; i<m_corpus.getFeatureSize(); i++){
				featureNameIndex.put(m_corpus.getFeature(i), featureNameIndex.size());
			}

			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(wordSimFileName), "UTF-8"));

			ArrayList<String> featureList = new ArrayList<String>();

			boolean firstLineFlag = false;
			int lineIndex = 0;
			while((tmpTxt=br.readLine())!=null){
				tmpTxt = tmpTxt.trim();
				if(tmpTxt.isEmpty())
					continue;

				lineContainer = tmpTxt.split("\t");
				if(firstLineFlag==false){
					for(int i=0; i<lineContainer.length; i++){
						featureList.add(lineContainer[i]);
					}
					firstLineFlag = true;
				}else{
					int rowWId = featureNameIndex.get(featureList.get(lineIndex));

					for(int i=0; i<lineContainer.length; i++){
						int colWId = featureNameIndex.get(featureList.get(i));
						m_wordSimMatrix[rowWId][colWId] = Double.parseDouble(lineContainer[i]);

						if(m_wordSimMatrix[rowWId][colWId] > maxSim){
							maxSim = m_wordSimMatrix[rowWId][colWId];
						}else{
							if(m_wordSimMatrix[rowWId][colWId] < minSim) {
								minSim = m_wordSimMatrix[rowWId][colWId];
							}
						}
					}

					lineIndex ++;
				}
			}
			normalizeSimByExp(simThreshold, vocabulary_size, m_wordSimMatrix, m_wordSimVec);
		}catch (Exception e){
			e.printStackTrace();
		}
	}

	public void normalizeSimByMaxMin(double maxSim, double minSim, double threshold, int vocabulary_size, double[][]m_wordSimMatrix, double[]m_wordSimVec){
		for(int i=0; i<vocabulary_size; i++) {
			m_wordSimVec[i] = 0;
			for (int j = 0; j < vocabulary_size; j++) {
				double normalizedSim = (m_wordSimMatrix[i][j] - minSim) / (maxSim - minSim);

				if(normalizedSim< threshold)
					m_wordSimMatrix[i][j] = 0;
				else
					m_wordSimMatrix[i][j] = normalizedSim;
				m_wordSimVec[i] += normalizedSim;
			}
		}
	}

	public void normalizeSimByExp(double threshold, int vocabulary_size, double[][]m_wordSimMatrix, double[]m_wordSimVec){
		for(int i=0; i<vocabulary_size; i++) {
			m_wordSimVec[i] = 0;
			for (int j = 0; j < vocabulary_size; j++) {
				double normalizedSim = Math.exp(m_wordSimMatrix[i][j] );

				if(normalizedSim< threshold)
					m_wordSimMatrix[i][j] = 0;
				else
					m_wordSimMatrix[i][j] = normalizedSim;
				m_wordSimVec[i] += normalizedSim;
			}
		}
	}

	public void randArticle(String filePrefix, String gloveFile){

		loadGloveVec(gloveFile);
		Random m_rand = new Random();

		ArrayList<_ParentDoc> pDocList = new ArrayList<_ParentDoc>();

		for(_Doc d: m_corpus.getCollection()){
			if(d instanceof _ParentDoc){
				if(d.getName().equals("444")){
					pDocList.add((_ParentDoc)d);
				}
			}
		}

		int randArticleIndex = m_rand.nextInt(pDocList.size());
		_ParentDoc pDoc = pDocList.get(randArticleIndex);

		try {
			double totalSim = 0;
			double avgSim = 0;
			double cosim =0;

			String parentSimFile = pDoc.getName()+".txt";
			PrintWriter parentOut = new PrintWriter(new File(
					filePrefix, parentSimFile));

			for (String feature:m_featureNames) {
				parentOut.print(feature + "\t");

				for(_SparseFeature pSf: pDoc.getSparse()) {
					int pSfVal = (int)pSf.getValue();
					String parentFeature = m_featureNames.get(pSf.getIndex());
					for(int i=0; i< pSfVal; i++)
						cosim = Utils.cosine(m_featureStat.get(feature).getM_gloveVec(), m_featureStat.get(parentFeature).getM_gloveVec());
						parentOut.print(parentFeature+":"+cosim+"\t");
				}

				parentOut.println();
			}
			parentOut.flush();
			parentOut.close();

		}catch (Exception e){
			e.printStackTrace();
		}

	}
}
