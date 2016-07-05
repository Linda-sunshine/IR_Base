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
	public static int ChildDocFeatureSize = 8;
	
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
		//get DF in child documents
		for(_Doc temp:m_trainSet) {
			if(temp instanceof _ChildDoc){
				_SparseFeature[] sfs = temp.getSparse();
				for(_SparseFeature sf : sfs){
					childDF[sf.getIndex()] ++;	// DF in child documents
					corpusDF[sf.getIndex()] ++;
				}
				childDocsNum += 1;
			}else{
				_SparseFeature[] sfs = temp.getSparse();
				for(_SparseFeature sf : sfs){
					parentDF[sf.getIndex()] ++;	// DF in child documents
					corpusDF[sf.getIndex()] ++;
				}
				parentDocsNum += 1;
			}
		}
		
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
						values[2] = IDFChild;
						values[3] = IDFChild==0 ? 0:IDFCorpus/IDFChild;
						
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
						
						values[4] = TFParent;//TF in parent document
						values[5] = TFChild;//TF in child document					
						values[6] = TFParent/TFChild;//TF ratio
						
						values[7] = IDFCorpus * TFParent;//TF-IDF
						w.setFeatures(values);
					}
				}
			
			}	
		}

	}
	
	
	public void EMonCorpus(){
		m_trainSet = m_corpus.getCollection();
		
		for(int i=0; i<number_of_iteration; i++){
			update_E_step();
			update_M_step(i);
		}
	}
	
	public void update_E_step(){
		super.EM();
	}
	
	public void update_M_step(int iter){
		for(_Doc d:m_trainSet){
			if(d instanceof _ParentDoc)
				updateFeatureWeight((_ParentDoc)d, iter);
		}
	}
	
	public void updateFeatureWeight(_ParentDoc pDoc, int iter){
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
		File modelFile = new File("model_"+iter);
		
		int featureNum = model.getNrFeature();
		for(int i=0; i<featureNum; i++)
			pDoc.m_featureWeight[i] = model.getDecfunCoef(i, 0);
		try{
			if((iter>200)&&(iter%100==0))
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
		if(xid==0)
			result = 1/(1+Math.exp(-result));
		else
			result = 1/(1+Math.exp(result));
		return result;
	}
}
