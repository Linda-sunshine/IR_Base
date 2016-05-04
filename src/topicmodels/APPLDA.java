package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._APPQuery;
import structures._ChildDoc;
import structures._ChildDoc4APP;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4APP;
import structures._RankItem;
import structures._SparseFeature;
import structures._Word;
import utils.Utils;

public class APPLDA extends ParentChildBase_Gibbs{
//	double d_alpha_parent; alpha
	double d_alpha_prior;
	double d_alpha_child;
	double d_alpha_child_off;
	double d_beta_child;
	double[] m_gamma;
	int m_number_of_topics_review;
	int m_totalTopics; 
	double[][] m_childTopicWordStats;
	double[][] m_childTopicWordProb;
	double[] m_childStat;
	double[][] m_xTopicProbCache;
	ArrayList<_APPQuery> m_APPQueries;
	
	public APPLDA(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics_description, int number_of_topics_review, double alpha, 
			double burnIn, int lag, double ksi, double tau, double alphaPrior, double alphaReview, 
			double alphaOff, double[] gamma, double betaReview, ArrayList<_APPQuery>queryList){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics_description, alpha, burnIn, lag, ksi, tau);
		
		d_alpha_child = alphaReview;
		d_alpha_prior  = alphaPrior;
		d_alpha_child_off = alphaOff;
		
		d_beta_child = betaReview;
		m_number_of_topics_review = number_of_topics_review;
		
		m_totalTopics = number_of_topics+m_number_of_topics_review;
		
		m_childStat = new double[m_number_of_topics_review];
		m_childTopicWordStats = new double[m_number_of_topics_review][vocabulary_size];
		m_childTopicWordProb = new double[m_number_of_topics_review][vocabulary_size];
		
		if (gamma!=null) {
			m_gamma = new double[gamma.length];
			System.arraycopy(gamma, 0, m_gamma, 0, gamma.length);
		}
		
		m_xTopicProbCache = new double[gamma.length][];
		m_xTopicProbCache[0] = new double[number_of_topics];
		m_xTopicProbCache[1] = new double[number_of_topics_review];

		m_kAlpha = d_alpha_prior*number_of_topics; //parent influence
		
		m_APPQueries = queryList;
	}
	
	public String toString(){
		return String.format("APP LDA [description topic size:%d, review topic size:%d, total topic size:%d, alpha^d:%.2f, alpha^r:%.2f, alpha^p:%.2f, alpha^off, beta:%.2f, beta_child:%.2f, gamma1:%.2f, gamma2:%.2f, training proportion:%.2f, Gibbs Sampling]", 
				number_of_topics, m_number_of_topics_review, m_totalTopics, d_alpha, d_alpha_child, d_alpha_prior, d_alpha_child_off, d_beta, m_gamma[0], m_gamma[1], m_testWord4PerplexityProportion);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		createSpace();
		
		for(int i=0; i<number_of_topics; i++){
			Arrays.fill(word_topic_sstat[i], d_beta);
		}
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(int i=0; i<m_number_of_topics_review; i++){
			Arrays.fill(m_childTopicWordStats[i], d_beta_child);
		}
		Arrays.fill(m_childStat, d_beta_child*vocabulary_size);
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				((_ParentDoc4APP)d).setTopics4Gibbs(number_of_topics, 0);
//				for(_Stn stnObj:d.getSentences())
//					stnObj.setTopic(number_of_topics);
			}else if(d instanceof _ChildDoc){
				((_ChildDoc4APP)d).createXSpace(number_of_topics, m_number_of_topics_review, m_gamma.length);
				((_ChildDoc4APP)d).setTopics4Gibbs(number_of_topics, m_number_of_topics_review, 0);
			}
			
			if(d instanceof _ParentDoc){
				for(_Word w:d.getWords()){
					word_topic_sstat[w.getTopic()][w.getIndex()] ++;
					m_sstat[w.getTopic()] ++;
				}
			}else{
				for(_Word w:d.getWords()){
					int xid = w.getX();
					if(xid==0){
						word_topic_sstat[w.getTopic()][w.getIndex()] ++;
						m_sstat[w.getTopic()] ++;
					}else{
						m_childTopicWordStats[w.getTopic()][w.getIndex()]++;
						m_childStat[w.getTopic()] ++;
					}
				}
			}
			
		}
		
		imposePrior();
		m_statisticsNormalized = false;
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc pDoc){
		// System.out.println("appLDA influence");
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		for(_ChildDoc cDoc: pDoc.m_childDocs){
			double muDp = number_of_topics*d_alpha_prior/(pDoc.getDocInferLength()+number_of_topics*d_alpha);
			term*= gammaFuncRatio((int)cDoc.m_xTopicSstat[0][tid], muDp, d_alpha_child+(pDoc.m_sstat[tid]+d_alpha)*muDp)
					/ gammaFuncRatio((int)cDoc.m_xTopicSstat[0][0], muDp, d_alpha_child+(pDoc.m_sstat[0]+d_alpha)*muDp);
		}
		
		return term;
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		_ChildDoc4APP cDoc = (_ChildDoc4APP) d;
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
				cDoc.m_xTopicSstat[xid][tid] --;
				cDoc.m_xSstat[xid] --;
				if(m_collectCorpusStats){
					m_childTopicWordStats[tid][wid] --;
					m_childStat[tid] --;
				}
			}
			
			normalizedProb = 0;
			double pLambdaZero = childXInDocProb(0, cDoc);
			for(tid=0; tid<number_of_topics; tid++){
				double pGlobalWordTopic = parentWordByTopicProb(tid, wid);
				double pGlobalTopicCDoc = globalChildTopicInDocProb(tid, cDoc);
				
				m_xTopicProbCache[0][tid] = pGlobalWordTopic*pGlobalTopicCDoc*pLambdaZero;
//				System.out.println("tid\t"+tid);
				normalizedProb += m_xTopicProbCache[0][tid];
			}
			
			double pLambdaOne = childXInDocProb(1, cDoc);
			for(tid=0; tid<m_number_of_topics_review; tid++){
				double pLocalWordTopic = childWordByTopicProb(tid, wid);
				double pLocalTopicCDoc = localChildTopicInDocProb(tid, cDoc);
				
				m_xTopicProbCache[1][tid] = pLocalWordTopic*pLocalTopicCDoc*pLambdaOne;
				normalizedProb += m_xTopicProbCache[1][tid];
			}
			
			boolean finishLoop = false;
			normalizedProb *= m_rand.nextDouble();
			
			xid = 0;
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_xTopicProbCache[0][tid];
				if(normalizedProb<=0){
					finishLoop = true;
					break;
				}
			}
				
			if(!finishLoop){	
				xid = 1;
				for(tid=0; tid<m_number_of_topics_review; tid++){
					normalizedProb -= m_xTopicProbCache[1][tid];
					if(normalizedProb<=0){
						finishLoop = true;
						break;
					}
				}	
			}
			
			if((tid==m_number_of_topics_review)&&(xid==1)){
				tid --;
			}
			
			w.setTopic(tid);
			w.setX(xid);
			
			cDoc.m_xTopicSstat[xid][tid] ++;
			cDoc.m_xSstat[xid] ++;

			if(xid==0){
				if (cDoc.m_wordXStat.containsKey(wid)) {
					cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid) + 1);
				} else {
					cDoc.m_wordXStat.put(wid, 1);
				}

				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] ++;
					m_sstat[tid] ++;
				}
			}else{
				if(m_collectCorpusStats){
					m_childTopicWordStats[tid][wid] ++;
					m_childStat[tid] ++;
				}
			}
		}
	}
	
	protected double childWordByTopicProb(int tid, int wid){
		return m_childTopicWordStats[tid][wid]/m_childStat[tid];
	}
	
	protected double globalChildTopicInDocProb(int tid, _ChildDoc4APP d){
		double parentInfluence = (d.m_parentDoc.m_sstat[tid]+d_alpha)/(d.m_parentDoc.getDocInferLength()+d_alpha*number_of_topics);
		
		double prob = (d.m_xTopicSstat[0][tid]+m_kAlpha*parentInfluence+d_alpha_child);
		prob /= (d.m_xSstat[0]+m_kAlpha+d_alpha_child*number_of_topics);
		
		return prob;
	}
	
	protected double localChildTopicInDocProb(int tid, _ChildDoc4APP d){
		
		double prob = (d.m_xTopicSstat[1][tid]+d_alpha_child_off);
		prob /= (d.m_xSstat[1]+d_alpha_child_off*number_of_topics);
		
		return prob;
	}
	
	protected double childXInDocProb(int xid, _ChildDoc4APP d){
		return m_gamma[xid] + d.m_xSstat[xid];
	}
	
	public void calculate_M_step(int iter){
		if(iter<m_burnIn && iter%m_lag==0){
			if (m_statisticsNormalized) {
				System.err.println("The statistics collector has been normlaized before, cannot further accumulate the samples!");
				System.exit(-1);
			}
			
			for(int i=0; i<number_of_topics; i++){
				for(int v=0; v<vocabulary_size; v++){
					topic_term_probabilty[i][v] += word_topic_sstat[i][v];
				}
			}
			
			for(int i=0; i<m_number_of_topics_review; i++){
				for(int v=0; v<vocabulary_size; v++){
					m_childTopicWordProb[i][v] += m_childTopicWordStats[i][v];
				}
			}
			
			for(_Doc d:m_trainSet){
				if(d instanceof _ParentDoc)
					collectParentStats((_ParentDoc)d);
				else if(d instanceof _ChildDoc)
					collectChildStats((_ChildDoc)d);
			}
		}
	}
	
	protected void collectParentStats(_ParentDoc d){
		for(int k=0; k<number_of_topics; k++){
			d.m_topics[k] += d.m_sstat[k] + d_alpha;
		}
	}
	
	protected void collectChildStats(_ChildDoc d){
		double parentInfluence = m_kAlpha/(d.m_parentDoc.getDocInferLength()+d_alpha*number_of_topics);
		
		for(int k=0; k<number_of_topics; k++){
			d.m_xTopics[0][k] += d.m_xTopicSstat[0][k]+parentInfluence*(d.m_parentDoc.m_sstat[k]+d_alpha)+d_alpha_child;
		}
		
		for(int k=0; k<m_number_of_topics_review; k++){
			d.m_xTopics[1][k] += d.m_xTopicSstat[1][k] + d_alpha_child_off;
		}
		
		for (int j = 0; j < m_gamma.length; j++)
			d.m_xProportion[j] += d.m_xSstat[j] + m_gamma[j];
		
		for(_Word w:d.getWords())
			w.collectXStats();
	}
	
	protected void estThetaInDoc(_Doc d){
		Utils.L1Normalization(d.m_topics);
//		if(d instanceof _ParentDoc){
//			estParentStnTopicProportion((_ParentDoc) d);
//		}else
		if(d instanceof _ChildDoc){
			((_ChildDoc)d).estGlobalLocalTheta();
		}
		
		m_statisticsNormalized = true;
	}
	
	protected double logLikelihoodByIntegrateTopics(_ParentDoc d){
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();
		
		for(int j=0; j<fv.length; j++){
			int wid = fv[j].getIndex();
			double value = fv[j].getValue();
			
			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*parentTopicInDocProb(k, d)/(d.getDocInferLength()+number_of_topics*d_alpha);
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
	
	protected double logLikelihoodByIntegrateTopics(_ChildDoc d){
		_ChildDoc4APP cDoc = (_ChildDoc4APP)d;
		double docLogLikelihood = 0.0;

		// prepare compute the normalizers
		_SparseFeature[] fv = d.getSparse();
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*globalChildTopicInDocProb(k, cDoc)*childXInDocProb(0, cDoc);
		
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			for(int k=0; k<m_number_of_topics_review; k++){
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*localChildTopicInDocProb(k, cDoc)*childXInDocProb(1, cDoc);
				
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
	
	protected void finalEst(){
		for(int i=0; i<this.number_of_topics; i++)
			Utils.L1Normalization(topic_term_probabilty[i]); 
		
		for(int i=0; i<m_number_of_topics_review; i++)
			Utils.L1Normalization(m_childTopicWordProb[i]);
		//estimate p(z|d) from all the collected samples
		for(_Doc d:m_trainSet)
			estThetaInDoc(d);
	}
	
	public void debugOutput(String filePrefix){

		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		if (!childTopicFolder.exists()) {
			System.out.println("creating directory" + childTopicFolder);
			childTopicFolder.mkdir();
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
				printParentTopicAssignment((_ParentDoc)d, parentTopicFolder);
//				printParentPhi((_ParentDoc)d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printChildTopicAssignment((_ChildDoc4APP)d, childTopicFolder);
				printXValue(d, childXFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParentParameter(parentParameterFile);
		printChildParameter(childParameterFile);
			
		printAPP4QueryByTopicModel(filePrefix);
		printAPP4QueryByLanguageModel(filePrefix);
		printAPP4QueryByHybrid(filePrefix);
		
		printReviewOnlyTopicWords(filePrefix);
	}
	
	protected void printReviewOnlyTopicWords(String filePrefix){
		int k = 20;
		String betaFile = filePrefix+"topWord4ReviewOnly.txt";
		try {
			System.out.println("beta file");
			PrintWriter betaOut = new PrintWriter(new File(betaFile));
			for (int i = 0; i < m_childTopicWordProb.length; i++) {
				MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
						k);
				for (int j = 0; j < vocabulary_size; j++)
					fVector.add(new _RankItem(m_corpus.getFeature(j),
							m_childTopicWordProb[i][j]));

//				betaOut.format("Topic %d(%.3f):\t", i, m_sstat[i]);
				for (_RankItem it : fVector) {
					betaOut.format("%s(%.3f)\t", it.m_name,
							m_logSpace ? Math.exp(it.m_value) : it.m_value);
					System.out.format("%s(%.3f)\t", it.m_name,
						m_logSpace ? Math.exp(it.m_value) : it.m_value);
				}
				betaOut.println();
				System.out.println();
			}
	
			betaOut.flush();
			betaOut.close();
		} catch (Exception ex) {
			System.err.print("File Not Found");
		}
	}
	
	protected void printParentParameter(String parentParameterFile){
		System.out.println("printing parent parameter");
		try{
			System.out.println(parentParameterFile);
			
			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			for(_Doc d: m_corpus.getCollection()){
				if(d instanceof _ParentDoc){
					parentParaOut.print(d.getName()+"\t");
					parentParaOut.print("topicProportion\t");
					for(int k=0; k<number_of_topics; k++){
						parentParaOut.print(d.m_topics[k]+"\t");
					}
					
					parentParaOut.println();
					
				}
			}
			
			parentParaOut.flush();
			parentParaOut.close();
			
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void printChildParameter(String childParameterFile){
		System.out.println("printing child parameter");
		try{
			System.out.println(childParameterFile);
			
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for(_Doc d: m_corpus.getCollection()){
	
				if(d instanceof _ChildDoc){
					childParaOut.print(d.getName()+"\t");
	
					childParaOut.print("shared topicProportion\t");
					for (int k = 0; k < number_of_topics; k++) {
						childParaOut.print(((_ChildDoc) d).m_xTopicSstat[0][k] + "\t");
					}

					childParaOut.print("off topicProportion\t");
					for (int k = 0; k < m_number_of_topics_review; k++) {
						childParaOut.print(((_ChildDoc) d).m_xTopicSstat[1][k] + "\t");
					}
					
					childParaOut.println();
				}
			
			}
			
			childParaOut.flush();
			childParaOut.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void printXValue(_Doc d, File childXFolder){
		String XValueFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(childXFolder,
					XValueFile));
	
			for(_Word w:d.getWords()){
				int index = w.getIndex();
				int x = w.getX();
				double xProb = w.getXProb();
				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + x + ":" + xProb + "\t");
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	protected void printChildTopicAssignment(_ChildDoc4APP d, File childFolder){
		String topicAssignmentFile = d.getName() + ".txt";
		try {

			PrintWriter pw = new PrintWriter(new File(childFolder, topicAssignmentFile));
			
			for (_Word w : d.getWords()) {
				int wid = w.getIndex();
				int topic = w.getTopic();
				int xid = w.getX();
				String featureName = m_corpus.getFeature(wid);
				pw.print(featureName + ":" + topic + ":"+ xid +"\t");
			
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	public void printParentTopicAssignment(_ParentDoc d, File parentFolder) {
		//	System.out.println("printing topic assignment parent documents");
			
		String topicAssignmentFile = d.getName() + ".txt";
		try {
			PrintWriter pw = new PrintWriter(new File(parentFolder,
					topicAssignmentFile));
				
			for(_Word w: d.getWords()){
				int index = w.getIndex();
				int topic = w.getTopic();
				String featureName = m_corpus.getFeature(index);
				pw.print(featureName + ":" + topic + "\t");
			}
			pw.println();
				
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	protected void printAPP4QueryByHybrid(String filePrefix){
		String topKChild4QueryFile = filePrefix + "topAPP4Query.txt";
		System.out.println("hybrid model rank");
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4QueryFile));
			
			for(_APPQuery appQuery: m_APPQueries){

				pw.print(appQuery.getQueryID()+"\t");
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc){
						double likelihood = rankAPP4QueryByHybrid(appQuery, (_ParentDoc)d);
						
						pw.print(d.getTitle()+":"+likelihood);
						pw.print("\t");
					}
				}
				
				pw.println();
			}
			
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected double rankAPP4QueryByHybrid(_APPQuery appQuery, _ParentDoc pDoc){
		double queryLikelihood = 0.0;
		
		double smoothingMu = m_LM.m_smoothingMu;
		_SparseFeature[] pFV = pDoc.getSparse();
		
		double docLenHybridVal = 0;
		docLenHybridVal += pDoc.getDocInferLength();
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			docLenHybridVal += cDoc.m_xSstat[0];
		}
		
		double alphaDoc = smoothingMu/(smoothingMu+docLenHybridVal);
		
		for(_Word w:appQuery.getWords()){
			int wid = w.getIndex();
			double wordLoglikelihood = 0;
			double featureHybridVal = 0;
		
			int featureIndex = Utils.indexOf(pFV, wid);
			double pDocVal = 0;
			if(featureIndex != -1){
				pDocVal = pFV[featureIndex].getValue();
			}
			featureHybridVal += pDocVal;
			
			for(_ChildDoc cDoc:pDoc.m_childDocs){
				_ChildDoc4APP cDoc4APP = (_ChildDoc4APP)cDoc;
				if (cDoc4APP.m_wordXStat.containsKey(wid)) {
					featureHybridVal += cDoc4APP.m_wordXStat.get(wid);
				}
			}
		
			double wordLMLikelihood = (1-alphaDoc)*(featureHybridVal/docLenHybridVal);
			
			wordLMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
			
			double wordTMLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*topicInHybridDocProb(k, pDoc);
				wordTMLikelihood += wordPerTopicLikelihood;
			}
			
			wordLoglikelihood = (m_tau)*wordLMLikelihood+(1-m_tau)*wordTMLikelihood;
			
			queryLikelihood += Math.log(wordLoglikelihood);
		}
		
		return queryLikelihood;
	}
	
	protected void printAPP4QueryByLanguageModel(String filePrefix){
		String topKChild4QueryFile = filePrefix + "topAPP4Query.txt";
		System.out.println("language model rank");
		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4QueryFile));
			
			m_LM.generateReferenceModelWithXVal();
			
			for(_APPQuery appQuery: m_APPQueries){
//				HashMap<String, Double> likelihoodMap = new HashMap<String, Double>();

				pw.print(appQuery.getQueryID()+"\t");
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc){
						double likelihood = rankAPP4QueryByLM(appQuery, (_ParentDoc)d);
//						likelihoodMap.put(d.getTitle(), likelihood);
						
						pw.print(d.getTitle()+":"+likelihood);
						pw.print("\t");
					}
				}
				
				pw.println();
			}
			
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected double rankAPP4QueryByLM(_APPQuery appQuery, _ParentDoc pDoc){
		double queryLikelihood = 0.0;
		
		double smoothingMu = m_LM.m_smoothingMu;
		_SparseFeature[] pFV = pDoc.getSparse();
		
		double docLenHybridVal = 0;
		docLenHybridVal += pDoc.getDocInferLength();
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			docLenHybridVal += cDoc.m_xSstat[0];
		}
		
		double alphaDoc = smoothingMu/(smoothingMu+docLenHybridVal);
		
		for(_Word w:appQuery.getWords()){
			int wid = w.getIndex();
			
			double featureHybridVal = 0;
		
			int featureIndex = Utils.indexOf(pFV, wid);
			double pDocVal = 0;
			if(featureIndex != -1){
				pDocVal = pFV[featureIndex].getValue();
			}
			featureHybridVal += pDocVal;
			
			for(_ChildDoc cDoc:pDoc.m_childDocs){
				_ChildDoc4APP cDoc4APP = (_ChildDoc4APP)cDoc;
				if (cDoc4APP.m_wordXStat.containsKey(wid)) {
					featureHybridVal += cDoc4APP.m_wordXStat.get(wid);
				}
			}
		
			double smoothingProb = (1-alphaDoc)*(featureHybridVal/docLenHybridVal);
			
			smoothingProb += alphaDoc*m_LM.getReferenceProb(wid);
				
			queryLikelihood += Math.log(smoothingProb);
		}
		
		return queryLikelihood;
	}
	
	protected void printAPP4QueryByTopicModel(String filePrefix){
		String topKChild4QueryFile = filePrefix + "topChild4Query.txt";
		System.out.println("topic model rank");

		try{
			PrintWriter pw = new PrintWriter(new File(topKChild4QueryFile));
			
			for(_APPQuery appQuery: m_APPQueries){
				
				pw.print(appQuery.getQueryID()+"\t");
				for(_Doc d:m_trainSet){
					if(d instanceof _ParentDoc){
						double likelihood = rankAPP4QueryByTM(appQuery, (_ParentDoc)d);
//						likelihoodMap.put(d.getTitle(), likelihood);
						
						pw.print(d.getTitle()+":"+likelihood);
						pw.print("\t");
					}
				}
				
				pw.println();
			}
			
			pw.flush();
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected double rankAPP4QueryByTM(_APPQuery query, _ParentDoc pDoc){
		double queryLikelihood = 0;
	
		for(_Word w:query.getWords()){
			int wid = w.getIndex();
				
			double wordLoglikelihood = 0;
				
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*topicInHybridDocProb(k, pDoc);
				wordLoglikelihood += wordPerTopicLikelihood;
			}
				queryLikelihood += Math.log(wordLoglikelihood);
			
		}
		
		return queryLikelihood;
	}
	
	protected double topicInHybridDocProb(int tid, _ParentDoc pDoc){
		
		double prob = 0;
		
		double topicNum = 0;
		double wordNum = 0;
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			topicNum += cDoc.m_xTopicSstat[0][tid];
			wordNum += cDoc.m_xSstat[0];
		}
		
		topicNum += pDoc.m_sstat[tid];
		wordNum += pDoc.getDocInferLength();		
		
		double parentInfluence = (pDoc.m_sstat[tid]+d_alpha)/(pDoc.getDocInferLength()+number_of_topics*d_alpha);
		prob = (d_alpha+topicNum+m_kAlpha*parentInfluence+d_alpha_child);
		
		prob /= (number_of_topics*d_alpha+wordNum+m_kAlpha+number_of_topics*d_alpha_child);
		
		return prob;
		
	}
}
