package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import com.sun.org.apache.xml.internal.resolver.helpers.PublicId;

import bsh.util.Util;
import structures._ChildDoc;
import structures._ChildDoc4ChildPhi;
import structures._ChildDoc4ThreePhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4ThreePhi;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import topicmodels.ParentChild_Gibbs.MatchPair;
import utils.Utils;

public class ParentChildWithChildPhi extends ParentChild_Gibbs{
	
	public double[] m_childTopicProbCache;
	public double[] m_gammaChild; // 3 dimensions in child
	
	public ParentChildWithChildPhi(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gammaChild, double mu) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gammaChild, mu);
		// TODO Auto-generated constructor stub
		
		m_topicProbCache = new double[number_of_topics];
		m_childTopicProbCache = new double[number_of_topics+1];
	
		m_gammaChild = new double[gammaChild.length];
		
		System.arraycopy(gammaChild, 0, m_gammaChild, 0, m_gammaChild.length);

	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection){
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size); // avoid adding such prior later on
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopicsVct(number_of_topics);				
			} else if(d instanceof _ChildDoc){
				((_ChildDoc4ChildPhi) d).createXSpace(number_of_topics, m_gammaChild.length);
				((_ChildDoc4ChildPhi) d).createLocalWordTopicDistribution(this.vocabulary_size, d_beta);
				computeMu4Doc((_ChildDoc4ChildPhi)d);
			}
			
			d.setTopics4Gibbs(number_of_topics, 0);
					
		}
		
		for(_Doc d:collection){
			if(d instanceof _ChildDoc4ChildPhi){
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
				
			}else if(d instanceof _ParentDoc){
				for(_Word w: d.getWords()){
					int tid = w.getTopic();
					int wid = w.getIndex();
					
					word_topic_sstat[tid][wid] ++;
					m_sstat[tid] ++;
				}
			}
			
		}
		
		imposePrior();
		
		m_statisticsNormalized = false;
	}
	
	protected void computeMu4Doc(_ChildDoc4ChildPhi d){
		_ParentDoc tempParent =  d.m_parentDoc;
		double mu = Utils.cosine_values(tempParent.getSparse(), d.getSparse());
		d.setMu(mu);
	}
	
	public void sampleInParentDoc(_ParentDoc d){
		int wid, tid, xid;
		double normalizedProb;
		
		for(_Word w: d.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			
			d.m_sstat[tid] --;
			
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0;
			
			double pWordTopic =0;
			for(tid=0; tid<number_of_topics; tid++){
				pWordTopic = parentWordByTopicProb(tid, wid);
				double pTopicPdoc = parentTopicInDocProb(tid, d);
				double pTopicCdoc = parentChildInfluenceProb(tid, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPdoc*pTopicCdoc;
				normalizedProb += m_topicProbCache[tid];
			}
	
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid == m_topicProbCache.length)
				tid --;
				
			w.setTopic(tid);
			d.m_sstat[tid] ++;

			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		
		}
	}
	

	protected double parentChildInfluenceProb(int tid, _ParentDoc d){
		double term = 1.0;
		double docLen = d.getTotalDocLength();
		
		if(tid==0)
			return term;
		
		for (_ChildDoc cDoc : d.m_childDocs) {
			double muDp =  cDoc.getMu()/docLen ;
			term *= gammaFuncRatio((int)cDoc.m_sstat[tid], muDp, d_alpha+d.m_sstat[tid]*muDp) 
					/ gammaFuncRatio((int)cDoc.m_sstat[0], muDp, d_alpha+d.m_sstat[0]*muDp);		
		} 

		return term;
	}

	protected void sampleInChildDoc(_ChildDoc d){
		_ChildDoc4ChildPhi doc = (_ChildDoc4ChildPhi)d;
		_ParentDoc pDoc = doc.m_parentDoc;
		int wid, tid, xid;
		
		double normalizedProb;
		for(_Word w: doc.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			if(xid==0){
				doc.m_sstat[tid] --;
				doc.m_globalWord --;
				
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] --;
					m_sstat[tid] --;
				}
			}else if(xid==1){
				doc.m_sstat[tid] --;
				doc.m_localWord --;
							
				doc.m_localTopicSstat --;
				doc.m_localWordTopicSstat[wid] --;
			}
			
			normalizedProb = 0;
			double pLambdaZero = childXInDocProb(0, doc);
			double pLambdaOne = childXInDocProb(1, doc);
			
			double pWordTopic = 0;
			for(tid=0; tid<number_of_topics; tid++){
				pWordTopic = childWordByTopicProb(tid, wid);
				
				double pTopic = childTopicInDocProb(tid, doc, pDoc);
				
				m_childTopicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
				normalizedProb += m_childTopicProbCache[tid];
			}
			
			
			pWordTopic = childLocalWordByTopicProb(wid, doc);
			m_childTopicProbCache[tid] = pWordTopic*pLambdaOne;
			normalizedProb += m_childTopicProbCache[tid];
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_childTopicProbCache.length; tid++){
				normalizedProb -= m_childTopicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==m_childTopicProbCache.length)
				tid --;
			
			if(tid<number_of_topics){
				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				doc.m_sstat[tid] ++;
				doc.m_globalWord ++;
				
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] ++;
					m_sstat[tid] ++;
 				}
				
			}else if(tid==number_of_topics){
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				doc.m_sstat[tid] ++;
				doc.m_localWord ++;
				
				doc.m_localTopicSstat++;
				doc.m_localWordTopicSstat[wid] ++;
			}
			
			
		}
	}
	
	protected double childXInDocProb(int xid, _ChildDoc4ChildPhi d){
		if(xid==0)
			return m_gammaChild[xid] + d.m_globalWord;
		else if(xid==1)
			return m_gammaChild[xid] + d.m_localWord;
		
		return 0;
	}
	
	protected double childLocalWordByTopicProb(int wid, _ChildDoc4ChildPhi d){
		return d.m_localWordTopicSstat[wid]/d.m_localTopicSstat;
	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc4ChildPhi cDoc, _ParentDoc pDoc){
		double docLength = pDoc.getTotalDocLength();
		
		return (d_alpha + cDoc.getMu()*pDoc.m_sstat[tid]/docLength + cDoc.m_sstat[tid])
					/(m_kAlpha + cDoc.getMu() + cDoc.m_globalWord);
		
	}
	
	protected void estThetaInDoc(_Doc d) {
		super.estThetaInDoc(d);
		if (d instanceof _ParentDoc){
			// estimate topic proportion of sentences in parent documents
			((_ParentDoc)d).estStnTheta();
//			((_ParentDoc)d).estGlobalLocalTheta();
		} else if (d instanceof _ChildDoc4ChildPhi) {
			((_ChildDoc4ChildPhi) d).estGlobalLocalTheta();
		}
		m_statisticsNormalized = true;
	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc)d;
		
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		pDoc.setTopics4Gibbs(number_of_topics, 0);		
		sampleTestSet.add(pDoc);
		
		
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			_ChildDoc4ChildPhi childDoc = (_ChildDoc4ChildPhi)cDoc; 
			childDoc.createXSpace(number_of_topics, m_gammaChild.length);
			childDoc.createLocalWordTopicDistribution(this.vocabulary_size, d_beta);
			computeMu4Doc(childDoc);
			
			cDoc.setTopics4Gibbs(number_of_topics, 0);
			sampleTestSet.add(childDoc);
			
		}
	}

	protected double calculate_log_likelihood(){
		double corpusLogLikelihood = 0;//how could we get the corpus-level likelihood?
		
		for(_Doc d: m_trainSet)
			corpusLogLikelihood += calculate_log_likelihood(d);
		
		return corpusLogLikelihood;
	}
	
	protected double logLikelihoodByIntegrateTopics(_ParentDoc doc) {
		_ParentDoc d = (_ParentDoc)doc;
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();
		double docLen = doc.getTotalDocLength();

		for (int j = 0; j < fv.length; j++) {
			int index = fv[j].getIndex();
			double value = fv[j].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = parentWordByTopicProb(k, index)*parentTopicInDocProb(k, d)/(docLen+number_of_topics*d_alpha);
				wordLogLikelihood += wordPerTopicLikelihood;
				
			}
						
			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += value * wordLogLikelihood;
		}

		return docLogLikelihood;
	}
	
	protected double logLikelihoodByIntegrateTopics(_ChildDoc doc) {
		_ChildDoc4ChildPhi d = (_ChildDoc4ChildPhi) doc;
		_ParentDoc pDoc = d.m_parentDoc;
		double docLogLikelihood = 0.0;
		double gammaLen = Utils.sumOfArray(m_gammaChild);
		// prepare compute the normalizers
		_SparseFeature[] fv = d.getSparse();
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();

			double wordLogLikelihood = 0;
			
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, d, pDoc)*childXInDocProb(0, d)/(d.getTotalDocLength()+gammaLen);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			
			wordLogLikelihood += childLocalWordByTopicProb(wid, d)*childXInDocProb(1, d)/(d.getTotalDocLength()+gammaLen);
			
			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}
			
			wordLogLikelihood = Math.log(wordLogLikelihood);
	
			docLogLikelihood += value * wordLogLikelihood;
		}
		
		return docLogLikelihood;
	}
	
	public void collectParentStats(_ParentDoc d){
		
		for(int k=0; k<this.number_of_topics; k++){
			d.m_topics[k] += d.m_sstat[k] + d_alpha;
		}
				
		d.collectTopicWordStat();

	}
	
	public void collectChildStats(_ChildDoc d){
		_ChildDoc4ChildPhi cDoc = (_ChildDoc4ChildPhi)d;
		for(int k=0; k<this.number_of_topics; k++){
			cDoc.m_topics[k] += cDoc.m_sstat[k] + d_alpha;
		}
		cDoc.m_topics[number_of_topics] += cDoc.m_sstat[number_of_topics];
		
		cDoc.collectLocalWordSstat();
	}

	public void estParentStnTopicProportion(_ParentDoc pDoc){
		return;
//		for(_Stn stnObj : pDoc.getSentences() ){
//			estStn(stnObj, pDoc);
//		}
	}
	
	public void estStn(_Stn stnObj,  _ParentDoc d){
		int i=0;
		initStn(stnObj);
		do{
			calculateStn_E_step(stnObj, d);
			if(i>m_burnIn && i%m_lag == 0){
				collectStnStats(stnObj, d);
			}
			
		}while(++i<number_of_iteration);
		
		Utils.L1Normalization(stnObj.m_topics);
	}
	
	public void initStn(_Stn stnObj){
		stnObj.setTopicsVct(number_of_topics);
	}
	
	public void calculateStn_E_step( _Stn stnObj, _ParentDoc d){
		stnObj.permuteStn();
		
		double normalizedProb = 0;
		int wid, tid;
		for(_Word w: stnObj.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			
			stnObj.m_topicSstat[tid] --;
			
			normalizedProb = 0;
			
			double pWordTopic = 0;
			
			for(tid=0; tid<number_of_topics; tid++){
				pWordTopic = parentWordByTopicProb(tid, wid);
				double pTopic = parentTopicInStnProb(tid, stnObj, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopic;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid --;
			
			
			w.setTopic(tid);
			stnObj.m_topicSstat[tid] ++;
	
			
		}
		
	}
	
	public double parentTopicInStnProb(int tid, _Stn stnObj, _ParentDoc d){
		return (d_alpha + d.m_topics[tid]+stnObj.m_topicSstat[tid])/(m_kAlpha+1+stnObj.getLength());
	}
	
	public void collectStnStats(_Stn stnObj, _ParentDoc d){
		for(int k=0; k<number_of_topics; k++){
			stnObj.m_topics[k] += stnObj.m_topicSstat[k]+d_alpha+d.m_topics[k];
		}
	}
	
	public void debugOutput(String filePrefix){

		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File parentPairTopicDistriFolder = new File(filePrefix+"pairTopic");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
		File childLocalWordTopicFolder = new File(filePrefix+ "childLocalTopic");
		
		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		if (!parentPairTopicDistriFolder.exists()) {
			System.out.println("creating pair directory" + parentPairTopicDistriFolder);
			parentPairTopicDistriFolder.mkdir();
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

		for (_Doc d : m_corpus.getCollection()) {
		if (d instanceof _ParentDoc) {
				printParentTopicAssignment(d, parentTopicFolder);
				printParentPhi((_ParentDoc)d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printTopicAssignment(d, childTopicFolder);
				printChildXValue(d, childXFolder);
				printChildLocalWordTopicDistribution((_ChildDoc4ChildPhi)d, childLocalWordTopicFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile);

		String similarityFile = filePrefix+"topicSimilarity.txt";
		discoverSpecificComments(MatchPair.MP_ChildDoc, similarityFile);
		
		printEntropy(filePrefix);
		
		int topKStn = 10;
		int topKChild = 10;
		
		printTopKChild4Stn(filePrefix, topKChild);
		
		printTopKStn4Child(filePrefix, topKStn);
		
		printTopKChild4Parent(filePrefix, topKChild);
	}
	
	public void printPairWordTopicDistribution(_ParentDoc4ThreePhi d, File parentPairTopicDistriFolder){
		String parentLocalTopicDistriFile = d.getName() + ".txt";
		try{			
			PrintWriter parentOut = new PrintWriter(new File(parentPairTopicDistriFolder, parentLocalTopicDistriFile));
			
			for(int wid=0; wid<this.vocabulary_size; wid++){
				String featureName = m_corpus.getFeature(wid);
				double wordTopicProb = d.m_pairWordTopicProb[wid];
				if(wordTopicProb > 0.001)
					parentOut.format("%s:%.3f\t", featureName, wordTopicProb);
			}
			parentOut.flush();
			parentOut.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void printChildLocalWordTopicDistribution(_ChildDoc4ChildPhi d, File childLocalTopicDistriFolder){
		
		String childLocalTopicDistriFile = d.getName() + ".txt";
		try{			
			PrintWriter childOut = new PrintWriter(new File(childLocalTopicDistriFolder, childLocalTopicDistriFile));
			
			for(int wid=0; wid<this.vocabulary_size; wid++){
				String featureName = m_corpus.getFeature(wid);
				double wordTopicProb = d.m_localWordTopicProb[wid];
				if(wordTopicProb > 0.001)
					childOut.format("%s:%.3f\t", featureName, wordTopicProb);
			}
			childOut.flush();
			childOut.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	public void printParentTopicAssignment(_Doc d, File parentFolder){
		String topicAssignmentFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(parentFolder,
					topicAssignmentFile));
			
//			for(int i=0; i<d.getSenetenceSize(); i++){
//				_Stn stnObj = d.getSentence(i);
//				for(_Word w: stnObj.getWords()){
//					int index = w.getIndex();
//					int topic = w.getTopic();
//					String featureName = m_corpus.getFeature(index);
//					pw.print(featureName + ":" + topic + "\t");
//				}
//				pw.println();
//			}
			
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	protected HashMap<Integer, Double> rankStn4ChildBySim( _ParentDoc pDoc, _ChildDoc cd){
		_ChildDoc4ChildPhi cDoc = (_ChildDoc4ChildPhi)cd;
		
		HashMap<Integer, Double> stnSimMap = new HashMap<Integer, Double>();
		
		double[] topics = new double[cDoc.m_topics.length-1];
		for(int i=0; i<cDoc.m_topics.length-1; i++){
			topics[i] = cDoc.m_topics[i];
		}
		
		Utils.L1Normalization(topics);
		
		for(_Stn stnObj:pDoc.getSentences()){
//			double stnSim = computeSimilarity(cDoc.m_topics, stnObj.m_topics);
//			stnSimMap.put(stnObj.getIndex()+1, stnSim);
			
//			double stnKL = Utils.klDivergence(topics, stnObj.m_topics);
			
			double stnKL = Utils.klDivergence(stnObj.m_topics, topics);
			stnSimMap.put(stnObj.getIndex()+1, -stnKL);
		}
		
		return stnSimMap;
	} 
	
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
		
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();
		double gammaLen = Utils.sumOfArray(m_gammaChild);
		
		for(_ChildDoc d:pDoc.m_childDocs){
			_ChildDoc4ChildPhi cDoc = (_ChildDoc4ChildPhi)d;
			int cDocLen = cDoc.getTotalDocLength();
			
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				int wid = w.getIndex();
			
				double wordLogLikelihood = 0;
				
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc, cDoc.m_parentDoc)*childXInDocProb(0, cDoc)/(cDoc.getTotalDocLength()+gammaLen);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				wordLogLikelihood += childLocalWordByTopicProb(wid, cDoc)*childXInDocProb(1, cDoc)/(cDoc.getTotalDocLength()+gammaLen);
								
				stnLogLikelihood += Math.log(wordLogLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
//				if(cDoc.m_stnLikelihoodMap.containsKey(stnObj.getIndex()))
//					stnLogLikelihood += cDoc.m_stnLikelihoodMap.get(stnObj.getIndex());
//				cDoc.m_stnLikelihoodMap.put(stnObj.getIndex(), stnLogLikelihood);
//			}	
	}
	
	protected double rankChild4ParentByLikelihood(_ParentDoc pDoc, _ChildDoc cd){
		_ChildDoc4ChildPhi cDoc = (_ChildDoc4ChildPhi) cd;
		int cDocLen = cDoc.getTotalDocLength();
		_SparseFeature[] fv = pDoc.getSparse();
		double gammaLen = Utils.sumOfArray(m_gammaChild);
		
		double docLogLikelihood = 0;
		for(_SparseFeature i: fv){
			int wid = i.getIndex();
			double value = i.getValue();
			
			double wordLogLikelihood = 0;
			
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, (_ChildDoc4ChildPhi)cDoc, pDoc)*childXInDocProb(0, (_ChildDoc4ChildPhi)cDoc)/(cDoc.getTotalDocLength()+gammaLen);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			wordLogLikelihood += childLocalWordByTopicProb(wid, (_ChildDoc4ChildPhi)cDoc)*childXInDocProb(1, (_ChildDoc4ChildPhi)cDoc)/(cDoc.getTotalDocLength()+gammaLen);
						
			docLogLikelihood += value*Math.log(wordLogLikelihood);
		}
	
		return docLogLikelihood;	
	}
	
	public void printParameter(String parentParameterFile, String childParameterFile){
		System.out.println("printing parameter");
		try{
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					parentParaOut.print(d.getName()+"\t");
					parentParaOut.print("topicProportion\t");
					for(int k=0; k<d.m_topics.length; k++){
						parentParaOut.print(d.m_topics[k]+"\t");
					}
					
					for(_Stn stnObj:d.getSentences()){							
						parentParaOut.print("sentence"+(stnObj.getIndex()+1)+"\t");
						for(int k=0; k<d.m_topics.length;k++){
							parentParaOut.print(stnObj.m_topics[k]+"\t");
						}
					}
					
					parentParaOut.println();
					
				}else{
					if(d instanceof _ChildDoc){
						childParaOut.print(d.getName()+"\t");

						childParaOut.print("topicProportion\t");
						for (int k = 0; k < d.m_topics.length; k++) {
							childParaOut.print(d.m_topics[k] + "\t");
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
}
