package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ACCTM_C extends ACCTM_TwoTheta{
	HashMap<Integer, Double> m_wordSstat;

	public ACCTM_C(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double ksi, double tau){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, ksi, tau);
	
		m_topicProbCache = new double[number_of_topics+1];
		m_wordSstat = new HashMap<Integer, Double>();
	}
	
	public String toString(){
		return String.format("Parent Child Base Phi^c topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		createSpace();
		
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				d.setTopics4Gibbs(number_of_topics, 0);
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopicsVct(number_of_topics);
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				((_ChildDoc4BaseWithPhi) d).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
				((_ChildDoc4BaseWithPhi) d).setTopics4Gibbs(number_of_topics, 0);
				computeMu4Doc((_ChildDoc) d);
			}
			
			if(d instanceof _ParentDoc){
				for (_Word w:d.getWords()) {
					word_topic_sstat[w.getTopic()][w.getIndex()]++;
					m_sstat[w.getTopic()]++;
				}	
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				for(_Word w: d.getWords()){
					int xid = w.getX();
					int tid = w.getTopic();
					int wid = w.getIndex();
					//update global
					if(xid==0){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
					}
				}
			}
		}
		
		imposePrior();	
		m_statisticsNormalized = false;
		generateLanguageModel();
	}
	
	protected void generateLanguageModel(){
		double totalWord = 0;
		
		for(_Doc d:m_corpus.getCollection()){
			if(d instanceof _ParentDoc)
				continue;
			_SparseFeature[] fv = d.getSparse();
			for(int i=0; i<fv.length; i++){
				int wid = fv[i].getIndex();
				double val = fv[i].getValue();
				
				totalWord += val;
				if(m_wordSstat.containsKey(wid)){
					double oldVal = m_wordSstat.get(wid);
					m_wordSstat.put(wid, oldVal+val);
				}else{
					m_wordSstat.put(wid, val);
				}
			}
		}
		
		for(int wid:m_wordSstat.keySet()){
			double val = m_wordSstat.get(wid);
			double prob = val/totalWord;
			m_wordSstat.put(wid, prob);
		}
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc pDoc){
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			double muDp = cDoc.getMu()/pDoc.getDocInferLength();
			term *= gammaFuncRatio((int)cDoc.m_xTopicSstat[0][tid], muDp, d_alpha+pDoc.m_sstat[tid]*muDp)
					/ gammaFuncRatio((int)cDoc.m_xTopicSstat[0][0], muDp, d_alpha+pDoc.m_sstat[0]*muDp);
		}
		
		return term;
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi)d;
		int wid, tid, xid;
		double normalizedProb;
		
		for(_Word w:cDoc.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			if(xid==0){
				cDoc.m_xTopicSstat[xid][tid] --;
				cDoc.m_xSstat[xid] --;
				cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid)-1);
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] --;
					m_sstat[tid] --;
				}
			}else if(xid==1){
				cDoc.m_xTopicSstat[xid][wid]--;
				cDoc.m_xSstat[xid] --;
				cDoc.m_childWordSstat --;
			}
			
			normalizedProb = 0;
			double pLambdaZero = childXInDocProb(0, cDoc);
			double pLambdaOne = childXInDocProb(1, cDoc);
			
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = childWordByTopicProb(tid, wid);
				double pTopic = childTopicInDocProb(tid, cDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
				normalizedProb += m_topicProbCache[tid];
			}
			
			double pWordTopic = childLocalWordByTopicProb(wid, cDoc);
			m_topicProbCache[tid] = pWordTopic*pLambdaOne;
			normalizedProb += m_topicProbCache[tid];
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid --;
			
			if(tid<number_of_topics){
				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				cDoc.m_xTopicSstat[xid][tid]++;
				cDoc.m_xSstat[xid]++;
				
				if(cDoc.m_wordXStat.containsKey(wid)){
					cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid)+1);
				}else{
					cDoc.m_wordXStat.put(wid, 1);
				}
				
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] ++;
					m_sstat[tid] ++;
 				}
				
			}else if(tid==(number_of_topics)){
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				cDoc.m_xTopicSstat[xid][wid]++;
				cDoc.m_xSstat[xid]++;
				cDoc.m_childWordSstat ++;
				
			}
		}
	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc d){
		double docLength = d.m_parentDoc.getDocInferLength();
		
		return (d_alpha + d.getMu()*d.m_parentDoc.m_sstat[tid]/docLength + d.m_xTopicSstat[0][tid])
					/(m_kAlpha + d.getMu() + d.m_xSstat[0]);
	}
	
	protected double childLocalWordByTopicProb(int wid, _ChildDoc4BaseWithPhi d){
		return (d.m_xTopicSstat[1][wid])
				/ (d.m_childWordSstat);
	}
	
	protected void collectChildStats(_ChildDoc d) {
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		_ParentDoc pDoc = cDoc.m_parentDoc;
		double parentDocLength = pDoc.getDocInferLength();
		
		for (int k = 0; k < this.number_of_topics; k++) 
			cDoc.m_xTopics[0][k] += cDoc.m_xTopicSstat[0][k] + d_alpha+cDoc.getMu()*pDoc.m_sstat[k] / parentDocLength;
		
		for(int x=0; x<m_gamma.length; x++)
			cDoc.m_xProportion[x] += m_gamma[x] + cDoc.m_xSstat[x];
		
		for(int w=0; w<vocabulary_size; w++)
			cDoc.m_xTopics[1][w] += cDoc.m_xTopicSstat[1][w];
	
		for(_Word w:d.getWords()){
			w.collectXStats();
		}
	}
	
	protected void estThetaInDoc(_Doc d) {
		
		if (d instanceof _ParentDoc) {
			((_ParentDoc)d).estStnTheta();
//			estParentStnTopicProportion((_ParentDoc) d);
			Utils.L1Normalization(d.m_topics);
		} else if (d instanceof _ChildDoc4BaseWithPhi) {
			((_ChildDoc4BaseWithPhi) d).estGlobalLocalTheta();
		}
		m_statisticsNormalized = true;
	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc)d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		////for conditional perplexity
		int testLength = 0; 
//		int testLength = (int) (m_testWord4PerplexityProportion*pDoc.getTotalDocLength());
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);		
		sampleTestSet.add(pDoc);
		
		pDoc.createSparseVct4Infer();
		
		for(_ChildDoc cDoc: pDoc.m_childDocs4Dynamic){
//		for(_ChildDoc cDoc: pDoc.m_childDocs){
			testLength =  (int) (m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
			((_ChildDoc4BaseWithPhi) cDoc).createXSpace(number_of_topics,
					m_gamma.length, vocabulary_size, d_beta);
			((_ChildDoc4BaseWithPhi)cDoc).setTopics4GibbsTest(number_of_topics, 0, testLength);
			sampleTestSet.add(cDoc);
			cDoc.createSparseVct4Infer();
			computeTestMu4Doc(cDoc);

		}
	}
	
	public void debugOutput(String filePrefix){

		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
		
		File childLocalWordTopicFolder = new File(filePrefix+ "childLocalTopic");

		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		if (!childTopicFolder.exists()) {
			System.out.println("creating directory" + childTopicFolder);
			childTopicFolder.mkdir();
		}
		if (!childLocalWordTopicFolder.exists()) {
			System.out.println("creating directory" + childLocalWordTopicFolder);
			childLocalWordTopicFolder.mkdir();
		}
		
		File parentPhiFolder = new File(filePrefix + "parentPhi");
		File childPhiFolder = new File(filePrefix + "childPhi");
		if (!parentPhiFolder.exists()) {
			System.out.println("creating directory" + parentPhiFolder);
			parentPhiFolder.mkdir();
		}
		if (!childPhiFolder.exists()) {
			System.out.println("creating directory" + childPhiFolder);
			childPhiFolder.mkdir();
		}
		
		File childXFolder = new File(filePrefix+"xValue");
		if(!childXFolder.exists()){
			System.out.println("creating x Value directory" + childXFolder);
			childXFolder.mkdir();
		}

		for (_Doc d : m_trainSet) {
		if (d instanceof _ParentDoc) {
				printParentTopicAssignment((_ParentDoc)d, parentTopicFolder);
				printParentPhi((_ParentDoc)d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printChildTopicAssignment(d, childTopicFolder);
				printChildLocalWordTopicDistribution((_ChildDoc4BaseWithPhi)d, childLocalWordTopicFolder);
				printXValue(d, childXFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile, m_trainSet);
		printTestParameter4Spam(filePrefix);
		String xProportionFile = filePrefix + "childXProportion.txt";
		printXProportion(xProportionFile, m_trainSet);
		
		String similarityFile = filePrefix+"topicSimilarity.txt";
		discoverSpecificComments(MatchPair.MP_ChildDoc, similarityFile);
		
		printEntropy(filePrefix);
		
		int topKStn = 10;
		int topKChild = 10;
		printTopKChild4Stn(filePrefix, topKChild);
//		printTopKChild4StnWithHybrid(filePrefix, topKChild);
		printTopKChild4StnWithHybridPro(filePrefix, topKChild);
		printTopKStn4Child(filePrefix, topKStn);
		
		printTopKChild4Parent(filePrefix, topKChild);
	}
	
	protected HashMap<Integer, Double> rankStn4ChildBySim( _ParentDoc pDoc, _ChildDoc cDoc){

		HashMap<Integer, Double> stnSimMap = new HashMap<Integer, Double>();
		
		for(_Stn stnObj:pDoc.getSentences()){
			double stnKL = Utils.klDivergence(cDoc.m_xTopics[0], stnObj.m_topics);

			stnSimMap.put(stnObj.getIndex()+1, -stnKL);
		}
		
		return stnSimMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByHybrid(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		double gammaLen = Utils.sumOfArray(m_gamma);
		
		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			double cDocLen = cDoc.getChildDocLenWithXVal();
						
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			for(_Word w:stnObj.getWords()){
				double featureLikelihood = 0;
				
				int wid = w.getIndex();
				
				double docVal = 0;
				if(cDoc.m_wordXStat.containsKey(wid)){
					docVal = cDoc.m_wordXStat.get(wid);
				}
				
				double LMLikelihood = (1-alphaDoc)*docVal/(cDocLen);
				
				LMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
				
				double TMLikelihood = 0;
				for(int k=0; k<number_of_topics; k++){
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					TMLikelihood += wordPerTopicLikelihood;
				}
				
//				TMLikelihood += childLocalWordByTopicProb(wid, (_ChildDoc4BaseWithPhi)cDoc)*childXInDocProb(1, cDoc)/ (cDoc.getTotalDocLength() + gammaLen);
				
				featureLikelihood = m_tau*LMLikelihood+(1-m_tau)*TMLikelihood;
				featureLikelihood = Math.log(featureLikelihood);
				stnLogLikelihood += featureLikelihood;
			}
		
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByHybridPro(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		double gammaLen = Utils.sumOfArray(m_gamma);
		
		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			double cDocLen = cDoc.getChildDocLenWithXVal();
						
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			for(_Word w:stnObj.getWords()){
				double featureLikelihood = 0;
				
				int wid = w.getIndex();
				
				double docVal = 0;
				if(cDoc.m_wordXStat.containsKey(wid)){
					docVal = cDoc.m_wordXStat.get(wid);
				}
				
				double LMLikelihood = (1-alphaDoc)*docVal/(cDocLen);
				
				LMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
				
				double TMLikelihood = 0;
				for(int k=0; k<number_of_topics; k++){
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					TMLikelihood += wordPerTopicLikelihood;
				}
				
//				TMLikelihood += childLocalWordByTopicProb(wid, (_ChildDoc4BaseWithPhi)cDoc)*childXInDocProb(1, cDoc)/ (cDoc.getTotalDocLength() + gammaLen);
				
				featureLikelihood = m_tau*LMLikelihood+(1-m_tau)*TMLikelihood;
				featureLikelihood = Math.log(featureLikelihood);
				stnLogLikelihood += featureLikelihood;
			}
		
			double cosineSim = computeSimilarity(stnObj.m_topics, cDoc.m_xTopics[0]);
			stnLogLikelihood = m_tau*stnLogLikelihood + (1-m_tau)*cosineSim;
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
		double gammaLen = Utils.sumOfArray(m_gamma);

		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();
		for(_ChildDoc d:pDoc.m_childDocs){
			_ChildDoc4BaseWithPhi cDoc =(_ChildDoc4BaseWithPhi)d;
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				int wid = w.getIndex();
			
				double wordLogLikelihood = 0;
				
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc)*childXInDocProb(0, cDoc)/(gammaLen+cDoc.getDocInferLength());
					wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				
				stnLogLikelihood += Math.log(wordLogLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected void printTopKChild4StnWithHybrid(String filePrefix, int topK){
		String topKChild4StnFile = filePrefix+"topChild4Stn_hybrid.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			m_LM.generateReferenceModelWithXVal();
			
			for(_Doc d: m_trainSet){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.getSenetenceSize());
					
					for(_Stn stnObj:pDoc.getSentences()){
						HashMap<String, Double> likelihoodMap = rankChild4StnByHybrid(stnObj, pDoc);
						
						pw.print((stnObj.getIndex()+1)+"\t");
						
						for(Map.Entry<String, Double> e: sortHashMap4String(likelihoodMap, true)){

							pw.print(e.getKey());
							pw.print(":"+e.getValue());
							pw.print("\t");
							
						}
						pw.println();		
				
					}
				}
			}
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	protected double logLikelihoodByIntegrateTopics(_ChildDoc d) {
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		double docLogLikelihood = 0.0;
		double gammaLen = Utils.sumOfArray(m_gamma);

		// prepare compute the normalizers
		_SparseFeature[] fv = cDoc.getSparse();
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc)*childXInDocProb(0, cDoc)/ (cDoc.getTotalDocLength() + gammaLen);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			double wordPerTopicLikelihood = childLocalWordByTopicProb(wid, cDoc)*childXInDocProb(1, cDoc)/ (cDoc.getTotalDocLength() + gammaLen);
			wordLogLikelihood += wordPerTopicLikelihood;

			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}
			
			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += value * wordLogLikelihood;
		}
		
		return docLogLikelihood;
	}
	
	public void printChildLocalWordTopicDistribution(_ChildDoc4BaseWithPhi d, File childLocalTopicDistriFolder){
		
		String childLocalTopicDistriFile = d.getName() + ".txt";
		try{			
			PrintWriter childOut = new PrintWriter(new File(childLocalTopicDistriFolder, childLocalTopicDistriFile));
			
			for(int wid=0; wid<this.vocabulary_size; wid++){
				String featureName = m_corpus.getFeature(wid);
				double wordTopicProb = d.m_xTopics[1][wid];
				if(wordTopicProb > 0.001)
					childOut.format("%s:%.3f\t", featureName, wordTopicProb);
			}
			childOut.flush();
			childOut.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
		
	}
		
	public void printXProportion(String xProportionFile, ArrayList<_Doc> docList){
		System.out.println("x proportion for parent doc");
		try{
			PrintWriter pw = new PrintWriter(new File(xProportionFile));
			for(_Doc d:docList){
				if(d instanceof _ParentDoc){
					for(_ChildDoc doc: ((_ParentDoc)d).m_childDocs){
						_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi)doc;
						pw.print(d.getName() + "\t");
						pw.print(cDoc.getName() + "\t");
						pw.print(cDoc.m_xProportion[0]+"\t");
						pw.print(cDoc.m_xProportion[1]);
						pw.println();
					}
				}
			}
			
			pw.flush();
			pw.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public void printXProportion4Dynamical(String xProportionFile, ArrayList<_Doc> docList){
		System.out.println("x proportion for parent doc");
		try{
			PrintWriter pw = new PrintWriter(new File(xProportionFile));
			for(_Doc d:docList){
				if(d instanceof _ParentDoc){
					for(_ChildDoc doc: ((_ParentDoc)d).m_childDocs4Dynamic){
						_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi)doc;
						pw.print(d.getName() + "\t");
						pw.print(cDoc.getName() + "\t");
						pw.print(cDoc.m_xProportion[0]+"\t");
						pw.print(cDoc.m_xProportion[1]);
						pw.println();
					}
				}
			}
			
			pw.flush();
			pw.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public void printParameter(String parentParameterFile, String childParameterFile, ArrayList<_Doc> docList){
		System.out.println("printing parameter");
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: docList){
				if(d instanceof _ParentDoc){
					parentParaOut.print(d.getName()+"\t");
					parentParaOut.print("topicProportion\t");
					for(int k=0; k<number_of_topics; k++){
						parentParaOut.print(d.m_topics[k]+"\t");
					}
					
					for(_Stn stnObj:d.getSentences()){							
						parentParaOut.print("sentence"+(stnObj.getIndex()+1)+"\t");
						for(int k=0; k<number_of_topics;k++){
							parentParaOut.print(stnObj.m_topics[k]+"\t");
						}
					}
					
					parentParaOut.println();
					
					for(_ChildDoc cDoc: ((_ParentDoc)d).m_childDocs){
						
						childParaOut.print(d.getName() + "\t");
						
						childParaOut.print(cDoc.getName()+"\t");
	
						childParaOut.print("topicProportion\t");
						for (int k = 0; k < number_of_topics; k++) {
							childParaOut.print(cDoc.m_xTopics[0][k] + "\t");
						}
						
						childParaOut.print("xProportion\t");
						for(int x=0; x<m_gamma.length; x++){
							childParaOut.print(cDoc.m_xProportion[x]+"\t");
						}
						
						childParaOut.println();		
					}
				}
			}
			
			parentParaOut.flush();
			parentParaOut.close();
			
			childParaOut.flush();
			childParaOut.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public void printParameter4Dynamical(String parentParameterFile, String childParameterFile, ArrayList<_Doc>docList){
		System.out.println("printing parameter");
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: docList){
				if(d instanceof _ParentDoc){
					parentParaOut.print(d.getName()+"\t");
					parentParaOut.print("topicProportion\t");
					for(int k=0; k<number_of_topics; k++){
						parentParaOut.print(d.m_topics[k]+"\t");
					}
					
					for(_Stn stnObj:d.getSentences()){							
						parentParaOut.print("sentence"+(stnObj.getIndex()+1)+"\t");
						for(int k=0; k<number_of_topics;k++){
							parentParaOut.print(stnObj.m_topics[k]+"\t");
						}
					}
					
					parentParaOut.println();
					
					for(_ChildDoc cDoc: ((_ParentDoc)d).m_childDocs4Dynamic){
						
						childParaOut.print(d.getName() + "\t");
						
						childParaOut.print(cDoc.getName()+"\t");
	
						childParaOut.print("topicProportion\t");
						for (int k = 0; k < number_of_topics; k++) {
							childParaOut.print(cDoc.m_xTopics[0][k] + "\t");
						}
						
						childParaOut.print("xProportion\t");
						for(int x=0; x<m_gamma.length; x++){
							childParaOut.print(cDoc.m_xProportion[x]+"\t");
						}
						
						childParaOut.println();		
					}
				}
			}
			
			parentParaOut.flush();
			parentParaOut.close();
			
			childParaOut.flush();
			childParaOut.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	protected double testLogLikelihoodByIntegrateTopics(_ChildDoc d) {
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		double docLogLikelihood = 0.0;
		double gammaLen = Utils.sumOfArray(m_gamma);

		// prepare compute the normalizers
		_SparseFeature[] fv = cDoc.getSparse();

		for (_Word w : cDoc.getTestWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double term1 = childWordByTopicProb(k, wid);
				double term2 = childTopicInDocProb(k, cDoc);
				double term3 = childXInDocProb(0, cDoc)/ (cDoc.getDocInferLength() + gammaLen);
				
				double wordPerTopicLikelihood = term1*term2*term3;
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			double wordPerTopicLikelihood = childLocalWordByTopicProb(wid, cDoc)
					* childXInDocProb(1, cDoc)
					/ (cDoc.getDocInferLength() + gammaLen);
			wordLogLikelihood += wordPerTopicLikelihood;

			if (Math.abs(wordLogLikelihood) < 1e-10) {
				System.out.println("wordLoglikelihood\t" + wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += wordLogLikelihood;
		}

		return docLogLikelihood;
	}
	
	public void initTest4Dynamical(ArrayList<_Doc> sampleTestSet, _Doc d, int commentNum){
		_ParentDoc pDoc = (_ParentDoc)d;
		pDoc.m_childDocs4Dynamic = new ArrayList<_ChildDoc>();
		pDoc.setTopics4Gibbs(number_of_topics, 0);
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}

		sampleTestSet.add(pDoc);
		int count = 0;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			if(count>=commentNum){
				break;
			}
			count ++;
			((_ChildDoc4BaseWithPhi)cDoc).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
			((_ChildDoc4BaseWithPhi)cDoc).setTopics4Gibbs(number_of_topics, 0);
			sampleTestSet.add(cDoc);
			pDoc.addChildDoc4Dynamics(cDoc);
			computeMu4Doc(cDoc);

		}
	}
	
	public void initTest4Spam(ArrayList<_Doc> sampleTestSet, _Doc d) {
		_ParentDoc pDoc = (_ParentDoc)d;
		pDoc.setTopics4Gibbs(number_of_topics, 0);
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}

		sampleTestSet.add(pDoc);
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			((_ChildDoc4BaseWithPhi)cDoc).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
			((_ChildDoc4BaseWithPhi)cDoc).setTopics4Gibbs(number_of_topics, 0);
			sampleTestSet.add(cDoc);
			cDoc.setParentDoc(pDoc);
			computeMu4Doc(cDoc);
		}
	}
	
	public void printTestParameter4Dynamic(int commentNum){
		String xProportionFile = "./data/results/dynamic/testChildXProportion_"+commentNum+".txt";
		printXProportion4Dynamical(xProportionFile, m_testSet);
		
		String parentParameterFile = "./data/results/dynamic/testParentParameter_"+commentNum+".txt";
		String childParameterFile = "./data/results/dynamic/testChildParameter_"+commentNum+".txt";
		printParameter4Dynamical(parentParameterFile, childParameterFile, m_testSet);
	}
	
	public void printTestParameter4Spam(String filePrefix){
		String xProportionFile = filePrefix+"testChildXProportion.txt";
		printXProportion(xProportionFile, m_testSet);
		
		String parentParameterFile = filePrefix+"testParentParameter.txt";
		String childParameterFile = filePrefix+"testChildParameter.txt";
		printParameter(parentParameterFile, childParameterFile, m_testSet);
	}
	
}
