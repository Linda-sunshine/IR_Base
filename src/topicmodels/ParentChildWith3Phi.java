package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Collections;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.List;
import structures._ChildDoc;
import structures._ChildDoc4OneTopicProportion;
import structures._ChildDoc4ThreePhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4ThreePhi;
import structures._SparseFeature;
import structures._Stn;
import structures._Stn4ThreePhi;
import structures._Word;
import topicmodels.ParentChild_Gibbs.MatchPair;
import util.Array;
import utils.Utils;

public class ParentChildWith3Phi extends ParentChild_Gibbs{

	public double[] m_childTopicProbCache;
	public double[] m_gammaParent; // 2 dimensions in parent
	public double[] m_gammaChild; // 3 dimensions in child
	
	public ParentChildWith3Phi(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gammaParent, double[] gammaChild, double mu) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gammaParent, mu);
		// TODO Auto-generated constructor stub
		
		m_topicProbCache = new double[number_of_topics+1];
		m_childTopicProbCache = new double[number_of_topics+2];
		
		m_gammaParent = new double[gammaParent.length];
		m_gammaChild = new double[gammaChild.length];
		
		System.arraycopy(gammaParent, 0, m_gammaParent, 0, m_gammaParent.length);
		System.arraycopy(gammaChild, 0, m_gammaChild, 0, m_gammaChild.length);

	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection){
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size); // avoid adding such prior later on
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				((_ParentDoc4ThreePhi)d).createXSpace(number_of_topics, m_gammaParent.length);
				((_ParentDoc4ThreePhi) d).createLocalWordTopicDistribution(this.vocabulary_size, d_beta);
				for(_Stn stnObj: d.getSentences())
					stnObj.setTopicsVct4ThreePhi(number_of_topics, m_gammaParent.length);				
			} else if(d instanceof _ChildDoc){
				((_ChildDoc4ThreePhi) d).createXSpace(number_of_topics, m_gammaChild.length);
				((_ChildDoc4ThreePhi) d).createLocalWordTopicDistribution(this.vocabulary_size, d_beta);
				computeMu4Doc((_ChildDoc4ThreePhi)d);
			}
			
			d.setTopics4Gibbs(number_of_topics, 0);
					
		}
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc4ThreePhi){
				for(_Word w: d.getWords()){
					int xid = w.getX();
					int tid = w.getTopic();
					int wid = w.getIndex();
					//update global
					if(xid==0){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
					}else{
					//update pair
						((_ParentDoc4ThreePhi)d).m_pairWordTopicSstat[wid] ++;
						((_ParentDoc4ThreePhi)d).m_pairWord ++;
					}
						
				}
			}else if(d instanceof _ChildDoc4ThreePhi){
				for(_Word w: d.getWords()){
					int xid = w.getX();
					int tid = w.getTopic();
					int wid = w.getIndex();
					//update global
					if(xid==0){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
					}else if(xid==1){
					//update pair
						_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi)((_ChildDoc4ThreePhi) d).m_parentDoc;
						(pDoc).m_pairWordTopicSstat[wid] ++;
						(pDoc).m_pairWord ++;
					}

				}
			}
		}
		
		imposePrior();
		
		m_statisticsNormalized = false;
	}
	
	protected void computeMu4Doc(_ChildDoc4ThreePhi d){
		_ParentDoc tempParent =  d.m_parentDoc;
		double mu = Utils.cosine_values(tempParent.getSparse(), d.getSparse());
		d.setMu(mu);
	}
	
	public void sampleInParentDoc(_ParentDoc doc){
		_ParentDoc4ThreePhi d = (_ParentDoc4ThreePhi)doc;
		int wid, tid, xid;
		double normalizedProb;
		double supplementProb;
		
		for(_Word w: d.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			if(xid==0){
				d.m_globalWord --;
				d.m_sstat[tid] --;
				
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] --;
					m_sstat[tid] --;
				}
			}else if(xid==1){
				d.m_sstat[tid] --;
				d.m_parentWord --;
				d.m_pairWord --;
				d.m_pairWordTopicSstat[wid] --;	
			}
			
			normalizedProb = 0;
			supplementProb = 0;
			
			double pLambdaZero = parentXInDocProb(0, d);
			double pLambdaOne = parentXInDocProb(1, d);
			
			double pWordTopic =0;
			for(tid=0; tid<number_of_topics; tid++){
				pWordTopic = globalParentWordByTopicProb(tid, wid);
				double pTopicPdoc = parentTopicInDocProb(tid, d);
				double pTopicCdoc = parentChildInfluenceProb(tid, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPdoc*pTopicCdoc*pLambdaZero;
				supplementProb += pTopicPdoc*pTopicCdoc;
			}
			
			for(tid=0; tid<number_of_topics; tid++){
				m_topicProbCache[tid] /= supplementProb;
				normalizedProb += m_topicProbCache[tid];
			}
			
			pWordTopic = localParentWordByTopicProb(wid, d);
			//extra one dimension
			m_topicProbCache[tid] = pWordTopic*pLambdaOne; 
			
			normalizedProb += m_topicProbCache[tid];
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid == m_topicProbCache.length)
				tid --;
			
			if(tid<number_of_topics){
				
				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				d.m_sstat[tid] ++;
				d.m_globalWord ++;
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] ++;
					m_sstat[tid] ++;
				}
			}else{
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				d.m_sstat[tid] ++;
				d.m_parentWord ++;
				d.m_pairWord ++;
				d.m_pairWordTopicSstat[wid] ++;
			}
				
		}
	}
	
	protected double globalParentWordByTopicProb(int tid, int wid){
		return word_topic_sstat[tid][wid]/m_sstat[tid];
	}
	
	//localword initialized vocabulary*beta;
	protected double localParentWordByTopicProb(int wid, _ParentDoc4ThreePhi d){
		return d.m_pairWordTopicSstat[wid]/d.m_pairWord;
	}
	
	protected double parentXInDocProb(int xid, _ParentDoc4ThreePhi d){
		if(xid ==0)
			return m_gammaParent[xid]+d.m_globalWord;
		else 
			return m_gammaParent[xid]+d.m_parentWord;
					
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc4ThreePhi d){
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		for (_ChildDoc cDoc : d.m_childDocs) {
			double muDp =  cDoc.getMu()/d.m_globalWord ;
			term *= gammaFuncRatio((int)cDoc.m_sstat[tid], muDp, d_alpha+d.m_sstat[tid]*muDp) 
					/ gammaFuncRatio((int)cDoc.m_sstat[0], muDp, d_alpha+d.m_sstat[0]*muDp);		
		} 

		return term;
	}

	protected void sampleInChildDoc(_ChildDoc d){
		_ChildDoc4ThreePhi doc = (_ChildDoc4ThreePhi)d;
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi) doc.m_parentDoc;
		int wid, tid, xid;
		
		double normalizedProb;
		for(_Word w: d.getWords()){
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
				doc.m_parentWord --;
				pDoc.m_pairWord --;
				pDoc.m_pairWordTopicSstat[wid] --;
			}else if(xid==2){
				doc.m_localWord --;
				doc.m_localTopicSstat --;
				doc.m_localWordTopicSstat[wid] --;
			}
			
			normalizedProb = 0;
			double pLambdaZero = childXInDocProb(0, doc);
			double pLambdaOne = childXInDocProb(1, doc);
			double pLambdaTwo = childXInDocProb(2, doc);
			
			double pWordTopic = 0;
			for(tid=0; tid<number_of_topics; tid++){
				pWordTopic = childWordByTopicProb(tid, wid);
				
				double pTopic = childTopicInDocProb(tid, doc, pDoc);
				
				m_childTopicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
				normalizedProb += m_childTopicProbCache[tid];
			}
			
			pWordTopic = childParentWordByTopicProb(wid, pDoc);
			m_childTopicProbCache[tid] = pWordTopic*pLambdaOne;
			normalizedProb += m_childTopicProbCache[tid];
			
			pWordTopic = childLocalWordByTopicProb(wid, doc);
			tid ++;
			m_childTopicProbCache[tid] = pWordTopic*pLambdaTwo;
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
				doc.m_parentWord ++;
				pDoc.m_pairWord ++;
				pDoc.m_pairWordTopicSstat[wid] ++;
			}else if(tid==(number_of_topics+1)){
				xid = 2;
				w.setX(xid);
				w.setTopic(tid);
				doc.m_localWord ++;
				doc.m_localTopicSstat++;
				doc.m_localWordTopicSstat[wid] ++;
			}
			
			
		}
	}
	
	protected double childXInDocProb(int xid, _ChildDoc4ThreePhi d){
		if(xid==0)
			return m_gammaChild[xid] + d.m_globalWord;
		else if(xid==1)
			return m_gammaChild[xid] + d.m_parentWord;
		else
			return m_gammaChild[xid] + d.m_localWord;
	}
	
	protected double childParentWordByTopicProb(int wid, _ParentDoc4ThreePhi pDoc){
		return pDoc.m_pairWordTopicSstat[wid]/pDoc.m_pairWord;
	}
	
	protected double childLocalWordByTopicProb(int wid, _ChildDoc4ThreePhi d){
		return d.m_localWordTopicSstat[wid]/d.m_localTopicSstat;
	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc4ThreePhi cDoc, _ParentDoc4ThreePhi pDoc){
		double docLength = pDoc.m_globalWord;
		
		return (d_alpha + cDoc.getMu()*pDoc.m_sstat[tid]/docLength + cDoc.m_sstat[tid])
					/(m_kAlpha + cDoc.getMu() + cDoc.m_globalWord);
		
	}
	
	protected void estThetaInDoc(_Doc d) {
		super.estThetaInDoc(d);
		if (d instanceof _ParentDoc4ThreePhi){
			// estimate topic proportion of sentences in parent documents
			((_ParentDoc4ThreePhi) d).estStnTheta();
			((_ParentDoc4ThreePhi) d).estGlobalLocalTheta();
		} else if (d instanceof _ChildDoc4ThreePhi) {
			((_ChildDoc4ThreePhi) d).estGlobalLocalTheta();
		}
		m_statisticsNormalized = true;
	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi)d;
		
		pDoc.createXSpace(number_of_topics, m_gammaParent.length);
		pDoc.createLocalWordTopicDistribution(this.vocabulary_size, d_beta);

		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct4ThreePhi(number_of_topics, m_gammaParent.length);
		}
		
		pDoc.setTopics4Gibbs(number_of_topics, 0);		
		sampleTestSet.add(pDoc);
		
		for(_Word w: pDoc.getWords()){
			int xid = w.getX();
			int tid = w.getTopic();
			int wid = w.getIndex();
			
			if(xid==1){
			//update pair
				pDoc.m_pairWordTopicSstat[wid] ++;
				pDoc.m_pairWord ++;
			}
		}
		
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			_ChildDoc4ThreePhi childDoc = (_ChildDoc4ThreePhi)cDoc; 
			childDoc.createXSpace(number_of_topics, m_gammaChild.length);
			childDoc.createLocalWordTopicDistribution(this.vocabulary_size, d_beta);
			computeMu4Doc(childDoc);
			
			cDoc.setTopics4Gibbs(number_of_topics, 0);
			sampleTestSet.add(childDoc);
			
			for(_Word w: childDoc.getWords()){
				int xid = w.getX();
				int tid = w.getTopic();
				int wid = w.getIndex();
				
				if(xid==1){
				//update pair
					pDoc.m_pairWordTopicSstat[wid] ++;
					pDoc.m_pairWord ++;
				}
			}
		}
	}
	
	protected double calculate_log_likelihood(){
		double corpusLogLikelihood = 0;//how could we get the corpus-level likelihood?
		
		for(_Doc d: m_trainSet)
			corpusLogLikelihood += calculate_log_likelihood(d);
		
		return corpusLogLikelihood;
	}
	
	protected double logLikelihoodByIntegrateTopics(_ParentDoc doc) {
		_ParentDoc4ThreePhi d = (_ParentDoc4ThreePhi)doc;
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();
		double gammaLen = Utils.sumOfArray(m_gammaParent);
		
		for (int j = 0; j < fv.length; j++) {
			int index = fv[j].getIndex();
			double value = fv[j].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = parentWordByTopicProb(k, index)*parentTopicInDocProb(k, d)/(d.m_globalWord+number_of_topics*d_alpha)*parentXInDocProb(0, d)/(d.getTotalDocLength()+gammaLen);
				wordLogLikelihood += wordPerTopicLikelihood;
				
			}
			
			wordLogLikelihood += localParentWordByTopicProb(index, d)*parentXInDocProb(1, d)/(d.getTotalDocLength()+gammaLen);
			
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
		_ChildDoc4ThreePhi d = (_ChildDoc4ThreePhi) doc;
		double docLogLikelihood = 0.0;

		// prepare compute the normalizers
		_SparseFeature[] fv = d.getSparse();
		double gammaLen = Utils.sumOfArray(m_gammaChild);
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();

			double wordLogLikelihood = 0;
			
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, d, (_ParentDoc4ThreePhi)d.m_parentDoc)*childXInDocProb(0, d)/(d.getTotalDocLength()+gammaLen);
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			wordLogLikelihood += childParentWordByTopicProb(wid, (_ParentDoc4ThreePhi)d.m_parentDoc)*childXInDocProb(1, d)/(d.getTotalDocLength()+gammaLen);
			
			wordLogLikelihood += childLocalWordByTopicProb(wid, d)*childXInDocProb(2, d)/(d.getTotalDocLength()+gammaLen);
			
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
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi)d;
		for(int k=0; k<this.number_of_topics; k++){
			pDoc.m_topics[k] += pDoc.m_sstat[k] + d_alpha;
		}
		pDoc.m_topics[number_of_topics] += pDoc.m_sstat[number_of_topics]; 
				
		pDoc.collectTopicWordStat();
		pDoc.collectLocalWordSstat();
	}
	
	public void collectChildStats(_ChildDoc d){
		_ChildDoc4ThreePhi cDoc = (_ChildDoc4ThreePhi)d;
		for(int k=0; k<this.number_of_topics; k++){
			cDoc.m_topics[k] += cDoc.m_sstat[k] + d_alpha;
		}
		cDoc.m_topics[number_of_topics] += cDoc.m_sstat[number_of_topics];
		
		cDoc.collectLocalWordSstat();
		
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi)cDoc.m_parentDoc;
//		rankStn4Child(cDoc, pDoc);
	}

	public void estParentStnTopicProportion(_ParentDoc pDoc){
		for(_Stn stnObj : pDoc.getSentences() ){
			estStn(stnObj, (_ParentDoc4ThreePhi)pDoc);
		}
	}
	
	public void estStn(_Stn stnObj,  _ParentDoc4ThreePhi d){
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
		stnObj.setTopicsVct4ThreePhi(number_of_topics, m_gammaParent.length);
	}
	
	public void calculateStn_E_step( _Stn stnObj, _ParentDoc4ThreePhi d){
		stnObj.permuteStn();
		
		double normalizedProb = 0;
		int wid, tid, xid;
		for(_Word w: stnObj.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			if(xid==0){
				stnObj.m_xSstat[xid] --;
				stnObj.m_topicSstat[tid] --;
			}else if(xid ==1){
				stnObj.m_topicSstat[tid] --;
				stnObj.m_xSstat[xid] --;
			}
			
			normalizedProb = 0;
			
			double pLambdaZero = parentXInStnProb(0, stnObj) ;
			double pLambdaOne = parentXInStnProb(1, stnObj);
			
			double pWordTopic = 0;
			
			for(tid=0; tid<number_of_topics; tid++){
				pWordTopic = globalParentWordByTopicProb(tid, wid);
				double pTopic = parentTopicInStnProb(tid, stnObj, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
				normalizedProb += m_topicProbCache[tid];
			}
			
			pWordTopic = localParentWordByTopicProb(wid, d);
			m_topicProbCache[tid] = pWordTopic*pLambdaOne;
			normalizedProb += m_topicProbCache[tid];
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid --;
			
			if(tid<number_of_topics){
				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				stnObj.m_topicSstat[tid] ++;
				stnObj.m_xSstat[xid] ++;
			}else{
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				stnObj.m_topicSstat[tid] ++;
				stnObj.m_xSstat[xid] ++;
			}
		}
		
	}
	
	public double parentXInStnProb(int xid, _Stn stnObj){
		return m_gammaParent[xid]+stnObj.m_xSstat[xid];
	}

	public double parentTopicInStnProb(int tid, _Stn stnObj, _ParentDoc4ThreePhi d){
		return (d_alpha + d.m_topics[tid]+stnObj.m_topicSstat[tid])/(m_kAlpha+1-d.m_topics[number_of_topics]+stnObj.m_xSstat[0]);
	}
	
	public void collectStnStats(_Stn stnObj, _ParentDoc4ThreePhi d){
		for(int k=0; k<number_of_topics; k++){
			stnObj.m_topics[k] += stnObj.m_topicSstat[k]+d_alpha+d.m_topics[k];
		}
		stnObj.m_topics[number_of_topics] += stnObj.m_topicSstat[number_of_topics];
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
		
		File childTopKStnFolder = new File(filePrefix+"topKStn");
		if(!childTopKStnFolder.exists()){
			System.out.println("creating top K stn directory\t"+childTopKStnFolder);
			childTopKStnFolder.mkdir();
		}
		
		File stnTopKChildFolder = new File(filePrefix+"topKChild");
		if(!stnTopKChildFolder.exists()){
			System.out.println("creating top K child directory\t"+stnTopKChildFolder);
			stnTopKChildFolder.mkdir();
		}
		
		int topKStn = 10;
		int topKChild = 10;
		for (_Doc d : m_corpus.getCollection()) {
			if (d instanceof _ParentDoc) {
				printTopicAssignment(d, parentTopicFolder);
				printParentPhi((_ParentDoc) d, parentPhiFolder);
				printPairWordTopicDistribution((_ParentDoc4ThreePhi) d, parentPairTopicDistriFolder);
			
//				printTopKChild4Stn(topKChild, (_ParentDoc4ThreePhi)d, stnTopKChildFolder);
//				printTopKStn4Child(topKStn, (_ParentDoc4ThreePhi)d, childTopKStnFolder);	
			} else if (d instanceof _ChildDoc) {
				printTopicAssignment(d, childTopicFolder);
				printChildXValue(d, childXFolder);
				printChildLocalWordTopicDistribution((_ChildDoc4ThreePhi) d, childLocalWordTopicFolder);
//				printTopKStn(topKStn, (_ChildDoc4ThreePhi) d, childTopKStnFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile);

		String similarityFile = filePrefix+"topicSimilarity.txt";
		discoverSpecificComments(MatchPair.MP_ChildDoc, similarityFile);
		
		printEntropy(filePrefix);
		
		printTopKChild4Stn(filePrefix, topKChild);
	}
	
	public void printParentPhi(_ParentDoc d, File phiFolder){
		String parentPhiFileName = d.getName()+".txt";
		_SparseFeature[] fv = d.getSparse();
		
		try{
			PrintWriter parentPW = new PrintWriter(new File(phiFolder, parentPhiFileName));
		
			for(int n=0; n<fv.length; n++){
				int index = fv[n].getIndex();
				String featureName = m_corpus.getFeature(index);
				parentPW.print(featureName + ":\t");
				for(int k=0; k<d.m_topics.length; k++)
					parentPW.print(d.m_phi[n][k]+"\t");
				parentPW.println();
			}
			parentPW.flush();
			parentPW.close();
		}catch(Exception ex){
			ex.printStackTrace();
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
	
	public void printChildLocalWordTopicDistribution(_ChildDoc4ThreePhi d, File childLocalTopicDistriFolder){
		
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
	
	void discoverSpecificComments(MatchPair matchType, String similarityFile) {
		System.out.println("topic similarity");
	
		try {
			PrintWriter pw = new PrintWriter(new File(similarityFile));

			for (_Doc doc : m_trainSet) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t");
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					for (_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
						pw.print(cDoc.getName() + ":");

						docTopicSimilarity = computeSimilarity(((_ParentDoc) doc).m_topics, cDoc.m_topics);
						pw.print(docTopicSimilarity);
						for (_Stn stnObj:doc.getSentences()) {
							if (matchType == MatchPair.MP_ChildDoc)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_topics);
							else if (matchType == MatchPair.MP_ChildGlobal)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_xTopics[0]);
							else if (matchType == MatchPair.MP_ChildLocal)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_xTopics[1]);
							
							pw.print(":"+(stnObj.getIndex()+1) + ":" + stnTopicSimilarity);
						}
						pw.print("\t");
					}
					pw.println();
				}
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
		
		//comment is a query, retrieve stn by topical similarity
	protected HashMap<Integer, Double> rankStn4ChildBySim( _ParentDoc4ThreePhi pDoc, _ChildDoc4ThreePhi cDoc){

		HashMap<Integer, Double> stnSimMap = new HashMap<Integer, Double>();
		
		for(_Stn stnObj:pDoc.getSentences()){
			double stnSim = computeSimilarity(cDoc.m_topics, stnObj.m_topics);
			stnSimMap.put(stnObj.getIndex()+1, stnSim);
		}
		
		return stnSimMap;
	}
	
		//stn is a query, retrieve comment by likelihood
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc4ThreePhi pDoc){
	
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();
		double gammaLen = Utils.sumOfArray(m_gammaChild);
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			int cDocLen = cDoc.getTotalDocLength();
			
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				int wid = w.getIndex();
			
				double wordLogLikelihood = 0;
				
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, (_ChildDoc4ThreePhi)cDoc, (_ParentDoc4ThreePhi)cDoc.m_parentDoc)*childXInDocProb(0, (_ChildDoc4ThreePhi)cDoc)/(cDoc.getTotalDocLength()+gammaLen);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				
				wordLogLikelihood += childParentWordByTopicProb(wid, (_ParentDoc4ThreePhi)cDoc.m_parentDoc)*childXInDocProb(1, (_ChildDoc4ThreePhi)cDoc)/(cDoc.getTotalDocLength()+gammaLen);
				
				wordLogLikelihood += childLocalWordByTopicProb(wid, (_ChildDoc4ThreePhi)cDoc)*childXInDocProb(2, (_ChildDoc4ThreePhi)cDoc)/(cDoc.getTotalDocLength()+gammaLen);
				
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
	
	protected void printTopKChild4Stn(String filePrefix, int topK){
		String topKChild4StnFile = filePrefix+"topChild4Stn.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));
			
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc4ThreePhi){
					_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi)d;
					
					pw.println(pDoc.getName()+"\t"+pDoc.getSenetenceSize());
					
					for(_Stn stnObj:pDoc.getSentences()){
						HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(stnObj, pDoc);
							
				
						int i=0;
						pw.print(stnObj.getIndex()+"\t");
						
						for(Map.Entry<String, Double> e: sortHashMap4String(likelihoodMap, true)){
							if(i==topK)
								break;
							pw.print(e.getKey());
							pw.print(":"+e.getValue());
							pw.print("\t");
							
							i++;
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
	
	protected void printTopKChild4Stn(int topK, _ParentDoc4ThreePhi pDoc, File topKChildFolder){
		File topKChild4PDocFolder = new File(topKChildFolder, pDoc.getName());
		if(!topKChild4PDocFolder.exists()){
//			System.out.println("creating top K stn directory\t"+topKChild4PDocFolder);
			topKChild4PDocFolder.mkdir();
		}
		
		for(_Stn stnObj:pDoc.getSentences()){
			HashMap<String, Double> likelihoodMap = rankChild4StnByLikelihood(stnObj, pDoc);
			String topChild4StnFile =  (stnObj.getIndex()+1)+".txt";
				
			try{
				int i=0;
				
				PrintWriter pw = new PrintWriter(new File(topKChild4PDocFolder, topChild4StnFile));
				
				for(Map.Entry<String, Double> e: sortHashMap4String(likelihoodMap, true)){
					if(i==topK)
						break;
					pw.print(e.getKey());
					pw.print("\t"+e.getValue());
					pw.println();
					
					i++;
				}
				
				pw.flush();
				pw.close();
			}catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	protected void printTopKStn4Child(int topK, _ParentDoc4ThreePhi pDoc, File topKStnFolder){
		File topKStn4PDocFolder = new File(topKStnFolder, pDoc.getName());
		if(!topKStn4PDocFolder.exists()){
//			System.out.println("creating top K stn directory\t"+topKStn4PDocFolder);
			topKStn4PDocFolder.mkdir();
		}
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			String topKStn4ChildFile = cDoc.getName()+".txt";
			HashMap<Integer, Double> stnSimMap = rankStn4ChildBySim(pDoc, (_ChildDoc4ThreePhi)cDoc);

			try{
				int i=0;
				
				PrintWriter pw = new PrintWriter(new File(topKStn4PDocFolder, topKStn4ChildFile));
				
				for(Map.Entry<Integer, Double> e: sortHashMap4Integer(stnSimMap, true)){
					if(i==topK)
						break;
					pw.print(e.getKey());
					pw.print("\t"+e.getValue());
					pw.println();
					
					i++;
				}
				
				pw.flush();
				pw.close();
			}catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	
//	public void rankStn4Child(_ChildDoc4ThreePhi cDoc, _ParentDoc4ThreePhi pDoc){
//
//		for(_Stn stnObj:pDoc.getSentences()){
//			
//			double stnLogLikelihood = 0;
//			for(_Word w: stnObj.getWords()){
//				double wordLikelihood = 0;
//				int wid = w.getIndex();
//			
//				for(int k=0; k<number_of_topics; k++){
//					wordLikelihood+=childXInDocProb(0, cDoc)*childTopicInDocProb(k, cDoc, pDoc)*childWordByTopicProb(k, wid);
//				}
//
//				wordLikelihood += childXInDocProb(1, cDoc)*childLocalWordByTopicProb(wid, cDoc);
//			
//				stnLogLikelihood += Math.log(wordLikelihood);
//			}
//			
//			if(cDoc.m_stnLikelihoodMap.containsKey(stnObj.getIndex()))
//				stnLogLikelihood += cDoc.m_stnLikelihoodMap.get(stnObj.getIndex());
//			cDoc.m_stnLikelihoodMap.put(stnObj.getIndex(), stnLogLikelihood);
//		}
//
//	}

	
	
}
