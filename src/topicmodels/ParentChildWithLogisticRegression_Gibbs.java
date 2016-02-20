package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import Analyzer.ParentChildAnalyzer;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._ChildDoc;
import structures._ChildDoc4LogisticRegression;
import structures._ChildDoc4ProbitModel;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Word;
import topicmodels.ParentChild_Gibbs.MatchPair;

public class ParentChildWithLogisticRegression_Gibbs extends ParentChild_Gibbs{

	public double[] m_lambda;
	
	public ParentChildWithLogisticRegression_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, mu);
		
		m_lambda = new double[ParentChildAnalyzer.ChildDocFeatureSize];
//		Arrays.fill(m_lambda, 0);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		super.initialize_probability(collection);
		Arrays.fill(m_lambda, 0);
	}
	
	void sampleInChildDoc(_ChildDoc doc){
		int wid, tid, xid;
		double normalizedProb;
		
		_ChildDoc4LogisticRegression d = (_ChildDoc4LogisticRegression) doc;
		
		_SparseFeature[] fv = d.getSparse();
	
		double[] xProb = new double[m_gamma.length];
		Arrays.fill(xProb, 0);
		
		for(_Word w: d.getWords()){
			normalizedProb = 0;
			xid = w.getX();
			tid = w.getTopic();
			
			d.m_xSstat[xid] --;
			d.m_xTopicSstat[xid][tid] --;
			
			double[] wordFeature = fv[w.getLocalIndex()].getValues();
			
			for(xid=0; xid<m_gamma.length; xid++){
				xProb[xid] = xPredictiveProb(xid, tid, wordFeature, d);
				normalizedProb += xProb[xid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(xid=0; xid<m_gamma.length; xid++){
				normalizedProb -= xProb[xid];
				if(normalizedProb<=0){
					break;
				}
			}
			
			if(xid == m_gamma.length)
				xid --;
			
			w.setX(xid);
			
			normalizedProb = 0;
			for(int i=0; i<m_gamma.length; i++)
				normalizedProb += xProb[i];
			w.setXValue(xProb[1]/normalizedProb);

			d.m_xSstat[xid] ++;
			d.m_xTopicSstat[xid][tid] ++;
			
		}
		
		
		for(_Word w:d.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
					
			d.m_xTopicSstat[xid][tid] --;
			d.m_xSstat[xid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0;
			
			for(tid=0; tid<number_of_topics; tid++){			
				m_topicProbCache[tid] = childWordByTopicProb(tid, wid)*childTopicInDocProb(tid, xid, d);
				normalizedProb += m_topicProbCache[tid];		
			}
		
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0){
						break;
					}
			}
			
			if (tid == number_of_topics)
				tid--;
			
			w.setTopic(tid);

			d.m_xTopicSstat[xid][tid] ++;
			d.m_xSstat[xid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid]++;
				m_sstat[tid]++;
			}	
					
		}
		
	}
	
	public double xPredictiveProb(int xid, int tid, double[] wordFeature, _ChildDoc4LogisticRegression d){
		double xProb = 0;
		
		xProb = childTopicInDocProb(tid, xid, d);
		xProb *= childXInDocProb(xid, wordFeature);
		return xProb;
	}
	
	public double childXInDocProb(int xid, double[] wordFeature){
		double xProb = 0;

		xProb = calLRPredict(m_lambda, wordFeature);
		xProb = (xid==1 ? xProb: (1-xProb));
//		System.out.println("xProb\t"+xProb);
		return xProb;
	}
	
	public void calculate_M_step(int iter){
//		System.out.print("iter\t"+iter);
		trainLogisticRegression();
		
		if(iter>m_burnIn && iter%m_lag == 0){
			if(m_statisticsNormalized){
				System.err.println("The statistics collector has been normlaized before, cannot further accumulate the samples!");
				System.exit(-1);
			}
			
			for(int i=0; i<this.number_of_topics; i++){
				for(int v=0; v<this.vocabulary_size; v++){
					topic_term_probabilty[i][v] += word_topic_sstat[i][v];
				}
			}
			
			for(_Doc d: m_trainSet){
				if(d instanceof _ParentDoc)
					collectParentStats((_ParentDoc)d);
				else if(d instanceof _ChildDoc)
					collectChildStats((_ChildDoc)d);
			}			
		}
		
	}
	
	double calLRPredict(double[]a, double[]b){
		double temp = utils.Utils.dotProduct(a, b);
		return 1/(1+Math.exp(-temp));	
	}
	
	public void trainLogisticRegression(){
//		System.out.println("train logistic regression");
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue = 0;
		int fSize = m_lambda.length;
		double[] g = new double[fSize];
		double[] diag = new double[fSize];
		
		Arrays.fill(g, 0);
		Arrays.fill(diag, 0);
		
		try{
			do{
				fValue = calcFuncGradient(g);
				LBFGS.lbfgs(fSize, 6, m_lambda, fValue, g, false, diag, iprint, 1e-2, 1e-20, iflag);
			}while(iflag[0] != 0);	
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
		
//		System.out.println("one...\t"+iflag[0]);
	
	}
	
	public double calcFuncGradient(double[] g){
		double fValue = 0;
		double ero = 0.01;
		
		double L2 = 0;
		double b = 0;
		for(int i=0; i<m_lambda.length; i++){
			b = m_lambda[i];
//			System.out.println("value\t"+i+"\t"+b);
			g[i] = 2*ero*b;
			L2 += b*b;
		}
		
		double Yi = 0;
		double predictYi = 0;
		_SparseFeature[] fv; 

		for(_Doc doc: m_trainSet){
			if(doc instanceof _ParentDoc)
				continue;
			fv = doc.getSparse(); 
			for(_Word w: doc.getWords()){
				Yi = w.getXValue();
//				Yi = w.getX();
				double[] wordFeature = fv[w.getLocalIndex()].getValues();
				predictYi = calLRPredict(wordFeature, m_lambda);
				fValue += Yi*Math.log(predictYi) + (1-Yi)*Math.log(1-predictYi);
				
				for(int i=0; i<m_lambda.length; i++){
					g[i] -= wordFeature[i]*(Yi-predictYi);
				}
			}
	
		}
	
		
		return ero*L2-fValue ;
		
	}
	
	public String toString(){
		return String.format("Parent Child topic model with logistic regression [k:%d, alpha:%.2f, beta:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta);
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
				printTopicAssignment(d, parentTopicFolder);
				printParentPhi((_ParentDoc)d, parentPhiFolder);
			} else if (d instanceof _ChildDoc) {
				printTopicAssignment(d, childTopicFolder);
				printChildXValue(d, childXFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile);

		String similarityFile = filePrefix+"topicSimilarity.txt";
		discoverSpecificComments(MatchPair.MP_ChildDoc, similarityFile);
		
		printEntropy(filePrefix);
		printLambdaValue(filePrefix);
		
	}

	public void printLambdaValue(String filePrefix){
		String lambdaFile = filePrefix + "lambda.txt";
		
		try{
			PrintWriter lambdaWriter = new PrintWriter(lambdaFile);
			
			for(int i=0; i<ParentChildAnalyzer.ChildDocFeatureSize; i++){
				lambdaWriter.print(m_lambda[i]+"\t");
			}
			
			lambdaWriter.println();
			lambdaWriter.flush();
			lambdaWriter.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
//	public double calExpectationPosteriorProb(_Word w, _ChildDoc4LogisticRegression d){
////		double posteriorProb = 0.0;
//		double expectation = 0.0;
//		int xid;
//		int tid;
//		double[] wordFeature = new double[ParentChildAnalyzer.ChildDocFeatureSize];
//		double[] xProb = new double[m_gamma.length];
//		double normalizedProb = 0;
//		
//		tid = w.getTopic();
//		xid = w.getX();
//		
//		for(int i=0; i<m_gamma.length; i++){
//			xProb[i] = xPredictiveProb(i, tid, wordFeature, d);
//			normalizedProb += xProb[i];
//		}
//		
//		for(int i=0; i<m_gamma.length; i++){
//			expectation += i*xProb[i]/normalizedProb;
//		}
//		
//		return expectation;
//	}

}
