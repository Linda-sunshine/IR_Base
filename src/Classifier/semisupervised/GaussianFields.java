package Classifier.semisupervised;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;
import Classifier.BaseClassifier;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.PRLogisticRegression;
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
	double[] m_cache; // cache the similarity computation results given the similarity metric is symmetric
	double[] m_fu; // predicted labels for unlabeled data.
	double[] m_Y; // true label for the labeled data and pseudo label from base learner
	SparseDoubleMatrix2D m_graph;
	
	MyPriorityQueue<_RankItem> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
	ArrayList<_Doc> m_labeled; // a subset of training set
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	
	BaseClassifier m_classifier; //Multiple learner.
	double[] m_pY;//p(Y), the probabilities of different classes.
	double[] m_pYSum; //\sum_i exp(-|c-fu(i)|)
	
	double m_discount = 0.5; // default similarity discount if across different products

	//Randomly pick 10% of all the training documents.
	public GaussianFields(_Corpus c, int classNumber, int featureSize, String classifier){
		super(c, classNumber, featureSize);
		
		m_labelRatio = 0.2;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 10000;
		m_k = 100;
		m_kPrime = 50;	
		m_labeled = new ArrayList<_Doc>();
		m_pY = new double[classNumber];
		m_pYSum = new double[classNumber];
		
		setClassifier(classifier);
	}	
	
	public GaussianFields(_Corpus c, int classNumber, int featureSize, String classifier, 
			double ratio, int k, int kPrime){
		super(c, classNumber, featureSize);
		
		m_labelRatio = ratio;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		m_labeled = new ArrayList<_Doc>();
		m_pY = new double[classNumber];
		m_pYSum = new double[classNumber];
		
		setClassifier(classifier);
	}
	
	@Override
	public String toString() {
		return String.format("Gaussian Fields with matrix inversion [C:%s, kUL:%d, kUU:%d, r:%.3f, alpha:%.3f, beta:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta);
	}
	
	private void setClassifier(String classifier) {
		if (classifier.equals("NB"))
			m_classifier = new NaiveBayes(null, m_classNo, m_featureSize);
		else if (classifier.equals("LR"))
			m_classifier = new LogisticRegression(null, m_classNo, m_featureSize);
		else if (classifier.equals("PR-LR"))
			m_classifier = new PRLogisticRegression(null, m_classNo, m_featureSize, 1.0);
		else if (classifier.equals("SVM"))
			m_classifier = new SVM(null, m_classNo, m_featureSize);
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
		
		//estimate the prior of p(y=c)
		Utils.scaleArray(m_pY, 1.0/Utils.sumOfArray(m_pY));
	}
	
	void initCache() {
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
		return (2*(m_U+m_L-1)-i)/2*(i+1) - ((m_U+m_L)-j);//specialized for the current matrix structure
	}
	
	void setCache(int i, int j, double v) {
		m_cache[encode(i,j)] = v;
	}
	
	double getCache(int i, int j) {
		return m_cache[encode(i,j)];
	}
	
	protected double getSimilarity(_Doc di, _Doc dj) {
		return Math.exp(Utils.calculateSimilarity(di, dj));
		//return Math.random();//just for debugging purpose
	}
	
	void constructGraph(boolean createSparseGraph) {
		double similarity = 0;
		m_L = m_labeled.size();
		m_U = m_testSet.size();
		
		/*** Set up cache structure for efficient computation. ****/
		initCache();
		if (m_fu==null || m_fu.length<m_U)
			m_fu = new double[m_U]; //otherwise we can reuse the current memory
		if (m_Y==null || m_Y.length<m_U+m_L)
			m_Y = new double[m_U+m_L];
		
		/*** pre-compute the full similarity matrix (except the diagonal). ****/
		_Doc di, dj;
		for (int i = 0; i < m_U; i++) {
			di = m_testSet.get(i);
			for (int j = i + 1; j < m_U; j++) {// to save computation since our similarity metric is symmetric
				dj = m_testSet.get(j);
				similarity = getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if (!di.sameProduct(dj))
					similarity *= m_discount;// differentiate reviews from different products
				setCache(i, j, similarity);
			}

			for (int j = 0; j < m_L; j++) {
				dj = m_labeled.get(j);
				similarity = getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if (!di.sameProduct(m_labeled.get(j)))
					similarity *= m_discount;// differentiate reviews from different products
				setCache(i, m_U + j, similarity);
			}
			
			//set up the Y vector for unlabeled data
			m_Y[i] = m_classifier.predict(m_testSet.get(i)); //Multiple learner.
		}	
		
		//set up the Y vector for labeled data
		for(int i=m_U; i<m_L+m_U; i++)
			m_Y[i] = m_labeled.get(i-m_U).getYLabel();
		
		/***Set up structure for k nearest neighbors.****/
		m_kUU = new MyPriorityQueue<_RankItem>(m_kPrime);
		m_kUL = new MyPriorityQueue<_RankItem>(m_k);
		
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
				m_graph.setQuick(i, n.m_index, scale * value);
				m_graph.setQuick(n.m_index, i, scale * value);
				sum += value;
			}
			m_kUU.clear();
			
			//Set the part of labeled and unlabeled nodes. L-U and U-L
			for(int j=0; j<m_L; j++) 
				m_kUL.add(new _RankItem(m_U+j, getCache(i,m_U+j)));
			
			for(_RankItem n:m_kUL) {
				value = Math.max(n.m_value, m_graph.getQuick(i, n.m_index)/scale);//recover the original Wij
				m_graph.setQuick(i, n.m_index, scale * value);
				m_graph.setQuick(n.m_index, i, scale * value);
				sum += value;
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
		
		System.out.println("Nearest neighbor graph construction finished!");
	}
	
	//Test the data set.
	@Override
	public void test(){	
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
		int pred, ans;
		for(int i = 0; i < m_U; i++) {
			pred = getLabel(m_fu[i]);
			ans = m_testSet.get(i).getYLabel();
			m_TPTable[pred][ans] += 1;
			if (m_debugOutput!=null && pred != ans)
				debug(m_testSet.get(i));
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
	}
	
	/**Different getLabel methods.**/
	//This is the original getLabel: -|c-p(c)|
	int getLabel(double pred) {
		for(int i=0; i<m_classNo; i++)
			m_cProbs[i] = -Math.abs(i-pred); //-|c-p(c)|
		return Utils.maxOfArrayIndex(m_cProbs);
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
		_RankItem item;
		_Doc neighbor;
		double sim, wijSumU=0, wijSumL=0;
		
		try {
			m_debugWriter.write(String.format("%d-%.4f(%d,%d):\t", d.getYLabel(), m_fu[id], getLabel(m_fu[id]), getLabel3(m_fu[id])));
		
			//find top five labeled
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(j, getCache(id, m_U + j)));
			
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL)
				wijSumL += n.m_value; //get the similarity between two nodes.
			
			/****Get the top 5 elements from kUL******/
			for(int k=0; k<5; k++){
				item = m_kUL.get(k);
				neighbor = m_labeled.get(item.m_index);
				sim = item.m_value/wijSumL;
				
				if (k==0)
					m_debugWriter.write(String.format("L[%d:%.4f, ", neighbor.getYLabel(), sim));
				else if (k==4)
					m_debugWriter.write(String.format("%d:%.4f]", neighbor.getYLabel(), sim));
				else
					m_debugWriter.write(String.format("%d:%.4f, ", neighbor.getYLabel(), sim));
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
			for(int k=0; k<5; k++){
				item = m_kUU.get(k);
				neighbor = m_testSet.get(item.m_index);
				sim = item.m_value/wijSumU;
				
				if (k==0)
					m_debugWriter.write(String.format("\tU[%.4f:%.4f, ", m_fu[neighbor.getID()], sim));
				else if (k==4)
					m_debugWriter.write(String.format("%.4f:%.4f]", m_fu[neighbor.getID()], sim));
				else
					m_debugWriter.write(String.format("%.4f:%.4f, ", m_fu[neighbor.getID()], sim));
			}
			m_kUU.clear();
			m_debugWriter.write("\n");		
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 
	
	@Override
	public int predict(_Doc doc) {
		return -1; //we don't support this in transductive learning
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation){
		
	}
}
