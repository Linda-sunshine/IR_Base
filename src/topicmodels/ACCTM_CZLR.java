package topicmodels;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;
import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import structures._stat;
import utils.Utils;

public class ACCTM_CZLR extends ACCTM_CZ{
//	protected double[] m_weight;
	public static int ChildDocFeatureSize = 6;
	
	public ACCTM_CZLR(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] weight, double ksi, double tau){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, weight, ksi, tau);
//		System.arraycopy(weight, 0, m_weight, 0, weight.length);
	}
	
	public String toString(){
		return String.format("ACCTM_CZLR topic model [k:%d, alpha:%.2f, beta:%.2f, Logistic regression Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		super.initialize_probability(collection);
		setFeatureValues();
	}
	
	public void setFeatureValues(){
		
		int N = m_trainSet.size(); // total number of documents
		int childDocsNum = 0;
		int parentDocsNum = 0;
		
		double[] childDF = new double[vocabulary_size]; // total number of unique words
		double[] corpusDF = new double[vocabulary_size];
		double[] parentDF = new double[vocabulary_size];

		double totalWords = 0;
		for(_Doc temp:m_trainSet) {
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
		for(_Doc tempDoc:m_trainSet) {	
			if(tempDoc instanceof _ParentDoc) {
				parentFvs = tempDoc.getSparse();
				_ParentDoc tempParentDoc = (_ParentDoc)tempDoc;
				tempParentDoc.initFeatureWeight(ChildDocFeatureSize);
				
				for(_ChildDoc tempChildDoc:((_ParentDoc) tempDoc).m_childDocs){
					_SparseFeature[] childFvs = tempChildDoc.getSparse();
					for(_Word w: tempChildDoc.getWords()){
						int wid = w.getIndex();
						
						double DFCorpus = corpusDF[wid];
						double IDFCorpus = DFCorpus>0 ? Math.log((N+1)/DFCorpus):0;
						
						double[] values = new double[ChildDocFeatureSize];
						
						double DFChild = childDF[wid];
						double IDFChild = DFChild>0 ? Math.log((childDocsNum+1)/DFChild):0;
						
						values[0] = 1;
						values[1] = IDFCorpus;
//						values[2] = IDFChild;
//						values[3] = IDFChild==0 ? 0:IDFCorpus/IDFChild;
						
						double TFParent = 0;
						double TFChild = 0;
						
						int wIndex = Utils.indexOf(parentFvs, wid);
						if(wIndex != -1){
							TFParent = parentFvs[wIndex].getValue();	
						}
						
						wIndex = Utils.indexOf(childFvs, wid);
						if(wIndex != -1){
							TFChild = childFvs[wIndex].getValue();	
						}
						
						values[2] = TFParent;//TF in parent document
						values[3] = TFChild;//TF in child document					
						values[4] = TFParent/TFChild;//TF ratio
						
						values[5] = IDFCorpus * TFChild;//TF-IDF
						w.setFeatures(values);
					}
				}
			
			}	
		}

	}
	
	public void EM() {
		System.out.format("Starting %s...\n", toString());
		
		long starttime = System.currentTimeMillis();
		
		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);
		
		String filePrefix = "./data/results/ACCTM_CZLR";
		File weightFolder = new File(filePrefix+"");
		if(!weightFolder.exists()){
//			System.out.println("creating directory for weight"+weightFolder);
			weightFolder.mkdir();
		}
		
		double delta=0, last=0, current=0;
		int i = 0, displayCount = 0;
		do {
			
			for(int j=0; j<number_of_iteration; j++){
				init();
				for(_Doc d:m_trainSet)
					calculate_E_step(d);
			}
			
			calculate_M_step(i, weightFolder);
			
			if (m_converge>0 || (m_displayLap>0 && i%m_displayLap==0 && displayCount > 6)){//required to display log-likelihood
				current = calculate_log_likelihood();//together with corpus-level log-likelihood
			
				if (i>0)
					delta = (last-current)/last;
				else
					delta = 1.0;
				last = current;
			}
			
			if (m_displayLap>0 && i%m_displayLap==0) {
				if (m_converge>0) {
					System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
					infoWriter.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
	
				} else {
					System.out.print(".");
					if (displayCount > 6){
						System.out.format("\t%d:%.3f\n", i, current);
						infoWriter.format("\t%d:%.3f\n", i, current);
					}
					displayCount ++;
				}
			}
			
			if (m_converge>0 && Math.abs(delta)<m_converge)
				break;//to speed-up, we don't need to compute likelihood in many cases
		} while (++i<this.number_of_iteration);
		
		finalEst();
		
		long endtime = System.currentTimeMillis() - starttime;
		System.out.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);	
		infoWriter.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);	
	}
	
	public void calculate_M_step(int iter, File weightFolder){
		update_M_step(iter, weightFolder);
	}
	
	public void update_E_step(){
		super.EM();
	}
	
	public void update_M_step(int iter, File weightFolder){

		if (m_statisticsNormalized) {
			System.err.println("The statistics collector has been normlaized before, cannot further accumulate the samples!");
			System.exit(-1);
		}
		
		for(int i=0; i<this.number_of_topics; i++){
			for(int v=0; v<this.vocabulary_size; v++){
				topic_term_probabilty[i][v] += word_topic_sstat[i][v];//collect the current sample
			}
		}
		
		// used to estimate final theta for each document
		for(_Doc d:m_trainSet){
			if(d instanceof _ParentDoc)
				collectParentStats((_ParentDoc)d);
			else if(d instanceof _ChildDoc)
				collectChildStats((_ChildDoc)d);
		}
		
		File weightIterFolder = new File(weightFolder, "_"+iter);
		if(!weightIterFolder.exists()){
			weightIterFolder.mkdir();
		}
		
		for(_Doc d:m_trainSet){
			if(d instanceof _ParentDoc)
				updateFeatureWeight((_ParentDoc)d, iter, weightIterFolder);
		}
	}	
	
	public void updateFeatureWeight(_ParentDoc pDoc, int iter, File weightIterFolder){
		int totalChildWordNum = 0;
		int featureLen = 0;
		ArrayList<Double> targetValList = new ArrayList<Double>();
		ArrayList<Feature[]> featureList = new ArrayList<Feature[]>();
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			for(_Word w:cDoc.getWords()){
				double[] wordFeatures = w.getFeatures();
				double x = w.getX();
				featureLen = wordFeatures.length;
				Feature[] featureVec = new Feature[featureLen];
				for(int i=0; i<featureLen; i++){
					featureVec[i] = new FeatureNode(i+1,wordFeatures[i]);
					
				}
				featureList.add(featureVec);
				targetValList.add(x);
			}
		}
		
		totalChildWordNum = featureList.size();
		double[] targetVal = new double[totalChildWordNum];
		Feature[][] featureMatrix = new Feature[totalChildWordNum][];
		for(int i=0; i<totalChildWordNum; i++){
			featureMatrix[i] =  featureList.get(i);
		}
		
		for(int i=0; i<totalChildWordNum; i++){
			targetVal[i] = targetValList.get(i);
		}
		
		Problem problem = new Problem();
		problem.l = totalChildWordNum;
		problem.n = featureLen+1;//featureNum
		problem.x = featureMatrix;
		problem.y = targetVal;
		
		SolverType solver = SolverType.L2R_LR;
		double C = 1.0;
		double eps = 0.01;
		Parameter param = new Parameter(solver, C, eps);
		Model model = Linear.train(problem, param);
		
		int featureNum = model.getNrFeature();
		for(int i=0; i<featureNum; i++)
			pDoc.m_featureWeight[i] = model.getDecfunCoef(i, 0);
		
		String weightFile = pDoc.getName()+".txt";
		File modelFile = new File(weightIterFolder, weightFile);
		try{
//			if((iter>200)&&(iter%100==0))
				model.save(modelFile);
		}catch(Exception e){
			System.out.println(e.getMessage());
		}
	}
	
	public void sampleInChildDoc(_ChildDoc d){
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
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
				cDoc.m_xTopicSstat[xid][wid] --;
				cDoc.m_xSstat[xid] --;
				cDoc.m_childWordSstat --;
			}
			
			normalizedProb = 0;
			double pLambdaZero = xProb4Word(0, w, cDoc);
			double pLambdaOne = xProb4Word(1, w, cDoc);
					
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
				
				cDoc.m_xTopicSstat[xid][tid] ++;
				cDoc.m_xSstat[xid] ++;
				
				if(cDoc.m_wordXStat.containsKey(wid)){
					cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid)+1);
				}else{
					cDoc.m_wordXStat.put(wid, 1);
				}
				
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid]++;
					m_sstat[tid] ++;
				}
			}else if(tid==(number_of_topics)){
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				
				cDoc.m_xTopicSstat[xid][wid] ++;
				cDoc.m_xSstat[xid] ++;
				cDoc.m_childWordSstat ++;
			}
		}
		
	}

	public double xProb4Word(int xid, _Word w, _ChildDoc cDoc){
		double result = 0;
		_ParentDoc pDoc = cDoc.m_parentDoc;
		result = Utils.dotProduct(pDoc.m_featureWeight, w.getFeatures());
		if(xid==1)
			result = 1/(1+Math.exp(-result));
		else
			result = 1/(1+Math.exp(result));
		return result;
	}

	protected double logLikelihoodByIntegrateTopics(_ChildDoc d){
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		double docLogLikelihood = 0;

		for(_Word w:d.getWords()){
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc)*xProb4Word(0, w, cDoc);
				wordLogLikelihood += wordPerTopicLikelihood;
			}

			double wordPerTopicLikelihood = childLocalWordByTopicProb(wid, cDoc)*xProb4Word(1, w, cDoc);
			wordLogLikelihood += wordPerTopicLikelihood;

			if(Math.abs(wordLogLikelihood)<1e-10){
				wordLogLikelihood += 1e-10;
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += wordLogLikelihood;
		}

		return docLogLikelihood;
	}	
}
