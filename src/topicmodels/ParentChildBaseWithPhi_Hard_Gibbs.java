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
import structures._ChildDoc4BaseWithPhi_Hard;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import topicmodels.ParentChild_Gibbs.MatchPair;
import utils.Utils;

public class ParentChildBaseWithPhi_Hard_Gibbs extends ParentChildBaseWithPhi_Gibbs{
	
	public double[] m_childTopicProbCache;
	
	public ParentChildBaseWithPhi_Hard_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu, double ksi, double tau){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, mu, ksi, tau);
	
		m_topicProbCache = new double[number_of_topics];
		m_childTopicProbCache = new double[number_of_topics+1];
		
	}
	
	public String toString(){
		return String.format("Parent Child Base Phi^c topic model with hard differentiate[k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				d.setTopics4Gibbs(number_of_topics, 0);
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopic(number_of_topics);
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				((_ChildDoc4BaseWithPhi_Hard) d).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
				((_ChildDoc4BaseWithPhi_Hard) d).setTopics4Gibbs(number_of_topics, 0);
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
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] --;
					m_sstat[tid] --;
				}
			}else if(xid==1){
				cDoc.m_xTopicSstat[xid][wid]--;
				cDoc.m_xSstat[xid] --;
				cDoc.m_childWordSstat --;
			}
			
			_ParentDoc pDocObj = cDoc.m_parentDoc;
			if(Utils.indexOf(pDocObj.getSparse(), wid) != -1){
				normalizedProb = 0;
				for(tid=0; tid<number_of_topics; tid++){
					double pWordTopic = childWordByTopicProb(tid, wid);
					double pTopic = childTopicInDocProb(tid, cDoc);
					
					m_topicProbCache[tid] = pWordTopic*pTopic;
					normalizedProb += m_topicProbCache[tid];
				}
				
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
					
					if(m_collectCorpusStats){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
	 				}
					
				}else if(tid==(number_of_topics)){
					System.out.println("error on hard differentiate");
					
				}
				
				
			}else{
				normalizedProb = 0;
				double pLambdaZero = childXInDocProb(0, cDoc);
				double pLambdaOne = childXInDocProb(1, cDoc);
				
				for(tid=0; tid<number_of_topics; tid++){
					double pWordTopic = childWordByTopicProb(tid, wid);
					double pTopic = childTopicInDocProb(tid, cDoc);
					
					m_childTopicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
					normalizedProb += m_childTopicProbCache[tid];
				}
					
				double pWordTopic = childLocalWordByTopicProb(wid, cDoc);
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
					cDoc.m_xTopicSstat[xid][tid]++;
					cDoc.m_xSstat[xid]++;
					
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
	}
	
	protected void estThetaInDoc(_Doc d) {
		
		if (d instanceof _ParentDoc) {
			// estimate topic proportion of sentences in parent documents
			// ((_ParentDoc4ThreePhi) d).estStnTheta();
			estParentStnTopicProportion((_ParentDoc) d);
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
		pDoc.setTopics4Gibbs(number_of_topics, 0);		
		sampleTestSet.add(pDoc);
		
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			((_ChildDoc4BaseWithPhi_Hard)cDoc).createXSpace(number_of_topics, m_gamma.length);
			((_ChildDoc4BaseWithPhi_Hard)cDoc).setTopics(number_of_topics, 0);
			sampleTestSet.add(cDoc);
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

		for (_Doc d : m_corpus.getCollection()) {
		if (d instanceof _ParentDoc) {
				printTopicAssignment((_ParentDoc)d, parentTopicFolder);
				printParentPhi((_ParentDoc)d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printTopicAssignment(d, childTopicFolder);
				printChildLocalWordTopicDistribution((_ChildDoc4BaseWithPhi)d, childLocalWordTopicFolder);
				printXValue(d, childXFolder);
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
	
	protected HashMap<Integer, Double> rankStn4ChildBySim( _ParentDoc pDoc, _ChildDoc cDoc){

		HashMap<Integer, Double> stnSimMap = new HashMap<Integer, Double>();
		
		for(_Stn stnObj:pDoc.getSentences()){
			double stnKL = Utils.klDivergence(cDoc.m_xTopics[0], stnObj.m_topics);

			stnSimMap.put(stnObj.getIndex()+1, -stnKL);
		}
		
		return stnSimMap;
	}
	
	protected HashMap<String , Double> rankChild4StnByLM(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();

		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			double cDocLen = cDoc.getChildDocLenWithXVal();
			_SparseFeature[] fv = cDoc.getSparse();
			
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			_SparseFeature[] sv = stnObj.getFv();
			for(_SparseFeature svWord:sv){
				double featureLikelihood = 0;
				
				int wid = svWord.getIndex();
				double stnVal = svWord.getValue();
				
				int featureIndex = Utils.indexOf(fv, wid);
				double docVal = 0;
				if(featureIndex!=-1){
					docVal = fv[featureIndex].getValue();
				}
				
				double smoothingProb = (1-alphaDoc)*docVal/(cDocLen);
				
				smoothingProb += alphaDoc*m_LM.getReferenceProb(wid);
				featureLikelihood = Math.log(smoothingProb);
				stnLogLikelihood += stnVal*featureLikelihood;
			}
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();
		
		for(_ChildDoc d:pDoc.m_childDocs){
			_ChildDoc4BaseWithPhi cDoc =(_ChildDoc4BaseWithPhi)d;
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				int wid = w.getIndex();
			
				double wordLogLikelihood = 0;
				
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				
				stnLogLikelihood += Math.log(wordLogLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected void printTopKChild4Stn(String filePrefix, int topK){
		String topKChild4StnFile = filePrefix+"topChild4Stn.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			m_LM.generateReferenceModelWithXVal();
			
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					_ParentDoc pDoc = (_ParentDoc)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.getSenetenceSize());
					
					for(_Stn stnObj:pDoc.getSentences()){
//						HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(stnObj, pDoc);
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
//		System.out.println("likelihood in child doc in base with phi");
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		double docLogLikelihood = 0.0;
		double gammaLen = Utils.sumOfArray(m_gamma);

		// prepare compute the normalizers
		_SparseFeature[] fv = cDoc.getSparse();
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();
			
			double wordLogLikelihood = 0;

			if(Utils.indexOf(d.m_parentDoc.getSparse(),wid)!=-1){
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
			}else{
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc)*childXInDocProb(0, cDoc)/ (cDoc.getTotalDocLength() + gammaLen);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				double wordPerTopicLikelihood = childLocalWordByTopicProb(wid, cDoc)*childXInDocProb(1, cDoc)/ (cDoc.getTotalDocLength() + gammaLen);
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
					
				}else{
					if(d instanceof _ChildDoc){
						childParaOut.print(d.getName()+"\t");

						childParaOut.print("topicProportion\t");
						for (int k = 0; k < number_of_topics; k++) {
							childParaOut.print((( _ChildDoc)d).m_xTopics[0][k] + "\t");
						}
						
						childParaOut.print("xProportion\t");
						for(int x=0; x<m_gamma.length; x++){
							childParaOut.print(((_ChildDoc)d).m_xProportion[x]+"\t");
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
