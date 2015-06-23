package Classifier.semisupervised;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._Clique;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import Classifier.BaseClassifier;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

public class GaussianFields extends BaseClassifier {
	
	double m_alpha; //Weight coefficient between unlabeled node and labeled node.
	double m_beta; //Weight coefficient between unlabeled node and unlabeled node.
	double m_M; //Influence of labeled node.
	int m_k; // k labeled nodes.
	int m_kPrime;//k' unlabeled nodes.
	
	int m_U, m_L;
	protected double[] m_cache; // cache the similarity computation results given the similarity metric is symmetric
	double[] m_fu; // predicted labels for unlabeled data.
	double[] m_Y; // true label for the labeled data and pseudo label from base learner
	SparseDoubleMatrix2D m_graph;
	
	MyPriorityQueue<_RankItem> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
	ArrayList<_Doc> m_labeled; // a subset of training set
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	
	BaseClassifier m_classifier; //Multiple learner.
	double[] m_pY;//p(Y), the probabilities of different classes.
	double[] m_pYSum; //\sum_i exp(-|c-fu(i)|)
	
	double m_discount = 1.0; // default similarity discount if across different products
	boolean m_outputPlot = false; // if we want to print out the curves for analysis
	Thread[] m_threadpool;

//	HashMap<Integer, String> m_IndexFeature;//For debug purpose.
	boolean m_topicFlag; //If it is true, then it will calculate the aspect similarity, otherwise, cosine similarity. 
	ArrayList<double[]> m_debugStat;
//	ArrayList<Double> m_posL, m_negL;
	//Randomly pick 10% of all the training documents.
	ArrayList<String> m_featureNames;//Store all features for debugging purpose.
	double[][] m_A; //The matrix used to store the result of metric learning.

	public GaussianFields(_Corpus c, String classifier, double C){
		super(c);
		
		m_labelRatio = 0.2;//an arbitrary setting
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 10000;
		m_k = 100;
		m_kPrime = 50;	
		m_labeled = new ArrayList<_Doc>();
		
		int classNumber = c.getClassSize();
		m_pY = new double[classNumber];
		m_pYSum = new double[classNumber];

		m_topicFlag = false;
		setClassifier(classifier, C);
//		m_IndexFeature = new HashMap<Integer, String>();
//		m_debugStat = new ArrayList<double[]>();
	}	
	
	public GaussianFields(_Corpus c, String classifier, double C, double ratio, int k, int kPrime){
		super(c);
		
		m_labelRatio = ratio;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		m_labeled = new ArrayList<_Doc>();
		
		int classNumber = c.getClassSize();
		m_pY = new double[classNumber];
		m_pYSum = new double[classNumber];

		m_topicFlag = false;
		setClassifier(classifier, C);
//		m_IndexFeature = new HashMap<Integer, String>();
		m_debugStat = new ArrayList<double[]>();
//		m_posL = new ArrayList<Double>();
//		m_negL = new ArrayList<Double>();
	}
	
	public void setTopicFlag(boolean a){
		m_topicFlag = a;
	}
	
	public String toString() {
		return String.format("Gaussian Fields with matrix inversion [C:%s, kUL:%d, kUU:%d, r:%.3f, alpha:%.3f, beta:%.3f, discount:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_discount);
	}
	
	private void setClassifier(String classifier, double C) {
		if (classifier.equals("NB"))
			m_classifier = new NaiveBayes(m_classNo, m_featureSize);
		else if (classifier.equals("LR"))
			m_classifier = new LogisticRegression(m_classNo, m_featureSize, C);
		else if (classifier.equals("SVM"))
			m_classifier = new SVM(m_classNo, m_featureSize, C);
		else {
			System.out.println("Classifier has not developed yet!");
			System.exit(-1);
		}
	}
	
	@Override
	protected void init() {
		m_labeled.clear();
		Arrays.fill(m_pY, 0);
		Arrays.fill(m_pYSum, 0);
	}
	
	//Train the data set.
	public void train(Collection<_Doc> trainSet){
		init();
		
		m_classifier.train(trainSet);
		
		//Randomly pick some training documents as the labeled documents.
		Random r = new Random();
		for (_Doc doc:trainSet){
			m_pY[doc.getYLabel()]++;
			if(r.nextDouble()<m_labelRatio)
				m_labeled.add(doc);
		}
		//Pick 1:1 data as the labeled cand
//		ArrayList<_Doc> neg = new ArrayList<_Doc>();
//		ArrayList<_Doc> pos = new ArrayList<_Doc>();
//		int negCount = 0, posCount = 0;
//		//Split the documents according to their labels.
//		for(_Doc doc: trainSet){
//			m_pY[doc.getYLabel()]++;
//			if(doc.getYLabel()==0)
//				neg.add(doc);
//			else 
//				pos.add(doc);
//		}
//		for(_Doc doc: neg){
//			if(r.nextDouble() < m_labelRatio){
//				m_labeled.add(doc);
//				negCount++;
//			}
//		}
//		for(_Doc doc: pos){
//			if(r.nextDouble() < m_labelRatio && posCount <= negCount){
//				m_labeled.add(doc);
//				posCount++;
//			}
//		}
		//estimate the prior of p(y=c)
		Utils.scaleArray(m_pY, 1.0/Utils.sumOfArray(m_pY));
	}
	
	protected void initCache() {
		int size = m_U*(2*m_L+m_U-1)/2;//specialized for the current matrix structure
		if (m_cache==null || m_cache.length<size)
			m_cache = new double[m_U*(2*m_L+m_U-1)/2]; // otherwise we can reuse the current memory space
	}
	
	int encode(int i, int j) {
		if (i>j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return i*(2*(m_U+m_L-1)-i+1)/2 + (j-i-1);//specialized for the current matrix structure
	}
	
	public void debugEncode() {
		m_U = 8; 
		m_L = 6;
		for(int i=0; i<m_U; i++) {
			for(int j=i+1; j<m_U; j++)
				System.out.print(encode(i,j) + " ");
			for(int j=0; j<m_L; j++)
				System.out.print(encode(i,m_U+j) + " ");
			System.out.println();
		}
	}
	
	public void setCache(int i, int j, double v) {
		m_cache[encode(i,j)] = v;
	}
	
	double getCache(int i, int j) {
		return m_cache[encode(i,j)];
	}
	
	public _Doc getTestDoc(int i) {
		return m_testSet.get(i);
	}
	
	public _Doc getLabeledDoc(int i) {
		return m_labeled.get(i);
	}
	
	public double getSimilarity(_Doc di, _Doc dj) {
		if (di.m_topics == null || dj.m_topics == null)
			return Math.exp(Utils.calculateSimilarity(di, dj));
		
		int topicSize = di.m_topics.length;
		double alpha = 1.0, beta = 1.0, gamma = 1.0;
		return Math.exp(alpha*Utils.calculateSimilarity(di, dj) - beta*Utils.KLsymmetric(di.m_topics, dj.m_topics)/topicSize);
//		if(m_A != null)
//			return Math.exp(alpha*Utils.calculateSimilarity(di, dj)) - beta*Utils.calculateMDistance(di, dj, m_A);
		
//		return Math.exp(Utils.calculateSimilarity(di, dj));//cosine similarity
//		return Math.exp(-Utils.KLsymmetric(di.m_topics, dj.m_topics)/topicSize);//topic similarity
//		return Math.exp(Utils.EuclideanDistance(di.m_sentiment, dj.m_sentiment)/topicSize);//sentiment similarity

//		return Math.exp(alpha*Utils.calculateSimilarity(di, dj) - beta*Utils.KLsymmetric(di.m_topics, dj.m_topics)/topicSize 
//				+ gamma*Utils.cosine(di.m_sentiment, dj.m_sentiment)/topicSize);
	}
	
	protected void calcSimilarityInThreads(){
		//using all the available CPUs!
		int cores = Runtime.getRuntime().availableProcessors();
		m_threadpool = new Thread[cores];
		int start = 0, end;
		double avgCost = (m_U * m_L + 0.5 * (m_U-1) * m_U)/cores, cost;
		System.out.format("Construct graph in parallel: L: %d, U: %d\n",  m_L, m_U);
		for(int i=0; i<cores; i++) {	
			if (i==cores-1)
				end = m_U;
			else {
				cost = avgCost;
				for(end = start; end<m_U && cost>=0; end++)
					cost -= m_L + (m_U-end-1);
			}
			
			m_threadpool[i] = new Thread(new PairwiseSimCalculator(this, start, end, m_topicFlag));
			
			start = end;
			m_threadpool[i].start();
		}
		
		for(int i=0; i<m_threadpool.length; i++){
			try {
				m_threadpool[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	void SimilarityCheck() {
		_Doc d, neighbor;
		int y;
		double[][][] prec = new double[3][2][2]; // p@5, p@10, p@20; p, n; U, L;
		double[][][] total = new double[3][2][2];
		double similiarty;
		
		try {
			BufferedWriter writer = null;
			if (m_outputPlot)
				writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./data/matlab/test.dat"), "UTF-8"));//fixed file location for curve plotting
				
			for(int i = 0; i < m_U; i++) {
				d = getTestDoc(i);
				y = d.getYLabel();
				
				/****Get the nearest neighbors of k'UU******/
				for (int j = 0; j < m_U; j++) {
					if (j == i)
						continue;
					similiarty = getCache(i, j);
					m_kUU.add(new _RankItem(j, similiarty));
					
					if (m_outputPlot && Math.random()<0.05)
						writer.write(String.format("%s %.5f\n", y==getTestDoc(j).getYLabel(), similiarty));
				}			
				
				int pos = 0;
				double precision = 0;
				for(_RankItem n: m_kUU){
					neighbor = getTestDoc(n.m_index);
					if (getLabel(m_fu[n.m_index]) == y)//prediction against the ground-truth
						precision ++;
					pos ++;
					
					if (pos==5) {
						prec[0][y][0] += precision/pos;
						total[0][y][0] ++;
					} else if (pos==10) {
						prec[1][y][0] += precision/pos;
						total[1][y][0] ++;
					} else if (pos==20) {
						prec[2][y][0] += precision/pos;
						total[2][y][0] ++;
						break;
					}
				}
				m_kUU.clear();
				
				/****Get the nearest neighbors of k'UL******/
				for (int j = 0; j < m_L; j++) {
					similiarty = getCache(i, m_U + j);
					m_kUL.add(new _RankItem(j, similiarty));
					if (m_outputPlot && Math.random()<0.05)
						writer.write(String.format("%s %.5f\n", y==getLabeledDoc(j).getYLabel(), similiarty));
				}
				
				precision = 0;
				pos = 0;
				for(_RankItem n: m_kUL){
					neighbor = getLabeledDoc(n.m_index);
					if (neighbor.getYLabel() == y)
						precision ++;
					pos ++;
					
					if (pos==5) {
						prec[0][y][1] += precision/pos;
						total[0][y][1] ++;
					} else if (pos==10) {
						prec[1][y][1] += precision/pos;
						total[1][y][1] ++;
					} else if (pos==20) {
						prec[2][y][1] += precision/pos;
						total[2][y][1] ++;
						break;
					}
				}
				m_kUL.clear();
			}
			
			if (writer!=null)
				writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("\nQuery\tDocs\tP@5\tP@10\tP@20");
		System.out.format("Pos\tU\t%.3f\t%.3f\t%.3f\n", prec[0][1][0]/total[0][1][0], prec[1][1][0]/total[1][1][0], prec[2][1][0]/total[2][1][0]);
		System.out.format("Pos\tL\t%.3f\t%.3f\t%.3f\n", prec[0][1][1]/total[0][1][1], prec[1][1][1]/total[1][1][1], prec[2][1][1]/total[2][1][1]);
		System.out.format("Neg\tU\t%.3f\t%.3f\t%.3f\n", prec[0][0][0]/total[0][0][0], prec[1][0][0]/total[1][0][0], prec[2][0][0]/total[2][0][0]);
		System.out.format("Neg\tL\t%.3f\t%.3f\t%.3f\n\n", prec[0][0][1]/total[0][0][1], prec[1][0][1]/total[1][0][1], prec[2][0][1]/total[2][0][1]);
	}
	
	protected void constructGraph(boolean createSparseGraph) {
		m_L = m_labeled.size();
		m_U = m_testSet.size();
		
		/*** Set up cache structure for efficient computation. ****/
		initCache();
		if (m_fu==null || m_fu.length<m_U)
			m_fu = new double[m_U]; //otherwise we can reuse the current memory
		if (m_Y==null || m_Y.length<m_U+m_L)
			m_Y = new double[m_U+m_L];
		
		/*** pre-compute the full similarity matrix (except the diagonal) in parallel. ****/
		calcSimilarityInThreads();
		
		//set up the Y vector for labeled data
		for(int i=m_U; i<m_L+m_U; i++)
			m_Y[i] = m_labeled.get(i-m_U).getYLabel();
		
		/***Set up structure for k nearest neighbors.****/
		m_kUU = new MyPriorityQueue<_RankItem>(m_kPrime);
		m_kUL = new MyPriorityQueue<_RankItem>(m_k);
		
		//temporary injected code
//		tmpSimilarityCheck();
		
		/***Set up document mapping for debugging purpose***/
		if (m_debugOutput!=null) {
			for (int i = 0; i < m_U; i++) 
				m_testSet.get(i).setID(i);//record the current position
		}
		
		if (!createSparseGraph) {
			System.out.println("Nearest neighbor graph construction finished!");
			return;//stop here if we want to save memory and construct the graph on the fly (space speed trade-off)
		}
		
		m_graph = new SparseDoubleMatrix2D(m_U+m_L, m_U+m_L);//we have to create this every time with exact dimension
		
		/****Construct the C+scale*\Delta matrix and Y vector.****/
		double scale = -m_alpha / (m_k + m_beta*m_kPrime), sum, value;
		int nz = 0;
		for(int i = 0; i < m_U; i++) {
			//set the part of unlabeled nodes. U-U
			for(int j=0; j<m_U; j++) {
				if (j==i)
					continue;
				
				m_kUU.add(new _RankItem(j, getCache(i,j)));
			}
			
			sum = 0;
			for(_RankItem n:m_kUU) {
				value = Math.max(m_beta*n.m_value, m_graph.getQuick(i, n.m_index)/scale);//recover the original Wij
				if (value!=0) {
					m_graph.setQuick(i, n.m_index, scale * value);
					m_graph.setQuick(n.m_index, i, scale * value);
					sum += value;
					nz ++;
				}
			}
			m_kUU.clear();
			
			//Set the part of labeled and unlabeled nodes. L-U and U-L
			for(int j=0; j<m_L; j++) 
				m_kUL.add(new _RankItem(m_U+j, getCache(i,m_U+j)));
			
			for(_RankItem n:m_kUL) {
				value = Math.max(n.m_value, m_graph.getQuick(i, n.m_index)/scale);//recover the original Wij
				if (value!=0) {
					m_graph.setQuick(i, n.m_index, scale * value);
					m_graph.setQuick(n.m_index, i, scale * value);
					sum += value;
					nz ++;
				}
			}
			m_graph.setQuick(i, i, 1-scale*sum);
			m_kUL.clear();
		}
		
		for(int i=m_U; i<m_L+m_U; i++) {
			sum = 0;
			for(int j=0; j<m_U; j++) 
				sum += m_graph.getQuick(i, j);
			m_graph.setQuick(i, i, m_M-sum); // scale has been already applied in each cell
		}
		
		System.out.format("Nearest neighbor graph (U[%d], L[%d], NZ[%d]) construction finished!\n", m_U, m_L, nz);
	}
	
	//Test the data set.
	@Override
	public double test(){	
		/***Construct the nearest neighbor graph****/
		constructGraph(true);
		
		/***Perform matrix inverse.****/
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		DoubleMatrix2D result = alg.inverse(m_graph);
		
		/***setting up the corresponding weight for the true labels***/
		for(int i=m_U; i<m_L+m_U; i++)
			m_Y[i] *= m_M;
		
		/***get some statistics***/
		for(int i = 0; i < m_U; i++){
			double pred = 0;
			for(int j=0; j<m_U+m_L; j++)
				pred += result.getQuick(i, j) * m_Y[j];			
			m_fu[i] = pred;//prediction for the unlabeled based on the labeled data and pseudo labels
			
			for(int j=0; j<m_classNo; j++)
				m_pYSum[j] += Math.exp(-Math.abs(j-m_fu[i]));			
		}
		
		/***evaluate the performance***/
		double acc = 0;
		int pred, ans;
		for(int i = 0; i < m_U; i++) {
			pred = getLabel(m_fu[i]);
			ans = m_testSet.get(i).getYLabel();
			m_TPTable[pred][ans] += 1;
			
			if (pred != ans) {
				if (m_debugOutput!=null)
					debug(m_testSet.get(i));
			} else 
				acc ++;
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		
		return acc/m_U;
	}
	
	/**Different getLabel methods.**/
	//This is the original getLabel: -|c-p(c)|
	int getLabel(double pred) {
		int label = 0;
		double minScore = Math.abs(pred), score;
		for(int i=1; i<m_classNo; i++) {
			score = Math.abs(i-pred);
			if (score<minScore) {
				minScore = score;
				label = i;
			}
		}
		return label;
	}
	
	//p(c) * exp(-|c-f(u_i)|)/sum_j{exp(-|c-f(u_j))} j represents all unlabeled data
	int getLabel3(double pred){
		for(int i = 0; i < m_classNo; i++)			
			m_cProbs[i] = m_pY[i] * Math.exp(-Math.abs(i-pred)) / m_pYSum[i];
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	//exp(-|c-f(u_i)|)/sum_j{exp(-|c-f(u_j))} j represents all unlabeled data, without class probabilities.
	int getLabel4(double pred) {		
		for (int i = 0; i < m_classNo; i++)
			m_cProbs[i] = Math.exp(-Math.abs(i - pred)) / m_pYSum[i];
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	@Override
	protected void debug(_Doc d){
		int id = d.getID();
		_SparseFeature[] dsfs = d.getSparse();
		_RankItem item;
		_Doc neighbor;
		double sim, wijSumU=0, wijSumL=0;
		
		try {
			m_debugWriter.write("============================================================================\n");
			m_debugWriter.write(String.format("Label:%d, prodID:%s, fu:%.4f, getLabel1:%d, getLabel3:%d, SVM:%d, Content:%s\n", d.getYLabel(), d.getItemID(), m_fu[id], getLabel(m_fu[id]), getLabel3(m_fu[id]), (int)m_Y[id], d.getSource()));
			
			for(int i = 0; i< dsfs.length; i++){
				String feature = m_featureNames.get(dsfs[i].getIndex());
				m_debugWriter.write(String.format("(%s %.4f),", feature, dsfs[i].getValue()));
			}
			m_debugWriter.write("\n");
			
			//find top five labeled
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++){
				m_kUL.add(new _RankItem(j, getCache(id, m_U + j)));
			}
	
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL)
				wijSumL += n.m_value; //get the similarity between two nodes.
			
			/****Get the top 5 elements from kUL******/
			m_debugWriter.write("*************************Labeled data*************************************\n");
			for(int k=0; k < 5; k++){
				item = m_kUL.get(k);
				neighbor = m_labeled.get(item.m_index);
				sim = item.m_value/wijSumL;
				
				//Print out the sparse vectors of the neighbors.
				m_debugWriter.write(String.format("Label:%d, prodID:%s, Similarity:%.4f\n", neighbor.getYLabel(), neighbor.getItemID(), sim));
				m_debugWriter.write(neighbor.getSource()+"\n");
				_SparseFeature[] sfs = neighbor.getSparse();
				int pointer1 = 0, pointer2 = 0;
				//Find out all the overlapping features and print them out.
				while(pointer1 < dsfs.length && pointer2 < sfs.length){
					_SparseFeature tmp1 = dsfs[pointer1];
					_SparseFeature tmp2 = sfs[pointer2];
					if(tmp1.getIndex() == tmp2.getIndex()){
						String feature = m_featureNames.get(tmp1.getIndex());
						m_debugWriter.write(String.format("(%s %.4f),", feature, tmp2.getValue()));
						pointer1++;
						pointer2++;
					} else if(tmp1.getIndex() < tmp2.getIndex())
						pointer1++;
					else pointer2++;
				}
				m_debugWriter.write("\n");
			}
			m_kUL.clear();
			
			//find top five unlabeled
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == id)
					continue;
				m_kUU.add(new _RankItem(j, getCache(id, j)));
			}
			
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU)
				wijSumU += n.m_value; //get the similarity between two nodes.
			
			/****Get the top 5 elements from k'UU******/
			m_debugWriter.write("*************************Unlabeled data*************************************\n");
			for(int k=0; k<5; k++){
				item = m_kUU.get(k);
				neighbor = m_testSet.get(item.m_index);
				sim = item.m_value/wijSumU;
				
				m_debugWriter.write(String.format("True Label:%d, prodID:%s, f_u:%.4f, Similarity:%.4f\n", neighbor.getYLabel(), neighbor.getItemID(), m_fu[neighbor.getID()], sim));
				m_debugWriter.write(neighbor.getSource()+"\n");
				_SparseFeature[] sfs = neighbor.getSparse();
				int pointer1 = 0, pointer2 = 0;
				//Find out all the overlapping features and print them out.
				while(pointer1 < dsfs.length && pointer2 < sfs.length){
					_SparseFeature tmp1 = dsfs[pointer1];
					_SparseFeature tmp2 = sfs[pointer2];
					if(tmp1.getIndex() == tmp2.getIndex()){
						String feature = m_featureNames.get(tmp1.getIndex());
						m_debugWriter.write(String.format("(%s %.4f),", feature, tmp2.getValue()));
						pointer1++;
						pointer2++;
					} else if(tmp1.getIndex() < tmp2.getIndex())
						pointer1++;
					else pointer2++;
				}
				m_debugWriter.write("\n");
			}
			m_kUU.clear();
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 	
	
	protected void calcPurity(_Doc d){
		double sameL = 0, sameU = 0;
		int id = d.getID();
		
		//Collect kUL and calculate purity.
		for (int j = 0; j < m_L; j++){
			m_kUL.add(new _RankItem(j, getCache(id, m_U + j)));
		}
		for(_RankItem n: m_kUL){
			int index = m_U + n.m_index;
			if(m_Y[index]==d.getYLabel())
				sameL++;
		}
		sameL = sameL / (m_kUL.size() + 0.0001);
		m_kUL.clear();
		
		//Collect kUU and calculate purity.
		for (int j = 0; j < m_U; j++) {
			if (j == id)
				continue;
			m_kUU.add(new _RankItem(j, getCache(id, j)));
		}
		for(_RankItem n: m_kUU){
			if(m_fu[n.m_index]==d.getYLabel())//Use fu as the comparison.
				sameU++;
		}
		sameU = sameU / (m_kUU.size() + 0.0001);
		m_kUU.clear();
		m_debugStat.add(new double[]{sameL, sameU});
	} 	
	
	public int predict(_Doc doc) {
		return -1; //we don't support this in transductive learning
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation) {
		
	}
	
//	public void printStat(){
//		double sumL = 0, sumU = 0;
//		for(int i =0 ; i < m_debugStat.size(); i++){
//			sumL += m_debugStat.get(i)[0];
//			sumU += m_debugStat.get(i)[1];
//		}
//		sumL = sumL / (m_debugStat.size() + 0.0001);
//		sumU = sumU / (m_debugStat.size() + 0.0001);
//		System.out.print(String.format("L&U purity\t%.4f\t%.4f\n", sumL, sumU));
//	}
	
	public void setMatrixA(double[][] A){
		this.m_A = A;
	}
	
	public void setFeatures(ArrayList<String> features){
		m_featureNames = features;
	}
}
