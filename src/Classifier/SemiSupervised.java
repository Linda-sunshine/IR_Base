package Classifier;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import utils.Utils;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

public class SemiSupervised extends BaseClassifier{
	class _Node implements Comparable<_Node> {
		int m_i, m_j; // index in the graph
		double m_sim; // similarity to the current node
		
		public _Node(int i, int j, double sim) {
			m_i = i;
			m_j = j;
			m_sim = sim;
		}

		@Override
		public int compareTo(_Node n) {
			if (m_sim<n.m_sim)
				return 1;
			else if (m_sim>n.m_sim)
				return -1;
			else
				return 0;
		}
	}
	
	protected double m_alpha; //Weight coefficient between unlabeled node and labeled node.
	protected double m_beta; //Weight coefficient between unlabeled node and unlabeled node.
	protected double m_M; //Influence of labeled node.
	protected int m_k; // k labeled nodes.
	protected int m_kPrime;//k' unlabeled nodes.
	
	protected MyPriorityQueue<_Node> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
	protected ArrayList<_Doc> m_labeled; // a subset of training set
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	
	protected BaseClassifier m_classifier; //Multiple learner.
	
	//Randomly pick 10% of all the training documents.
	public SemiSupervised(_Corpus c, int classNumber, int featureSize, String classifier){
		super(c, classNumber, featureSize);
		
		m_labelRatio = 0.1;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 100000;
		m_k = 100;
		m_kPrime = 50;	
		m_labeled = new ArrayList<_Doc>();
		
		setClassifier(classifier);
	}	
	
	public SemiSupervised(_Corpus c, int classNumber, int featureSize, String classifier, 
			double ratio, int k, int kPrime){
		super(c, classNumber, featureSize);
		
		m_labelRatio = ratio;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 100000;
		m_k = k;
		m_kPrime = kPrime;	
		m_labeled = new ArrayList<_Doc>();
		
		setClassifier(classifier);
	}
	
	private void setClassifier(String classifier) {
		if (classifier.equals("NB"))
			m_classifier = new NaiveBayes(null, m_classNo, m_featureSize);
		else if (classifier.equals("LR"))
			m_classifier = new LogisticRegression(null, m_classNo, m_featureSize);
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
	}
	
	//Train the data set.
	public void train(Collection<_Doc> trainSet){	
		init();
		
		m_classifier.train(trainSet);
		
		//Randomly pick some training documents as the labeled documents.
		Random r = new Random();
		for (_Doc doc:trainSet){
			if(r.nextDouble()<m_labelRatio){
				m_labeled.add(doc);
			}
		}
	}
	
	//Test the data set.
	public void test(){
		double similarity = 0;
		int L = m_labeled.size(), U = m_testSet.size();
		double[][] Wij = new double[U+L][U+L];
		
		/***Set up structure for k nearest neighbors.****/
		m_kUU = new MyPriorityQueue<_Node>(m_kPrime);
		m_kUL = new MyPriorityQueue<_Node>(m_k);
		
		/***Construct the Wij matrix.****/
		for(int i = 0; i < U; i++){
			//set the part of unlabeled nodes. U-U
			for(int j = 0; j < i; j++){//not including i-self
				similarity = m_beta * Utils.calculateSimilarity(m_testSet.get(i), m_testSet.get(j));
				m_kUU.add(new _Node(i, j, similarity));
			}	
			
			for(_Node n:m_kUU) {
				Wij[n.m_i][n.m_j] = n.m_sim;
				Wij[n.m_j][n.m_i] = n.m_sim;
			}
			m_kUU.clear();
			
			//Set the part of labeled and unlabeled nodes. L-U and U-L
			for(int j = 0; j < L; j++){
				similarity = m_alpha * Utils.calculateSimilarity(m_testSet.get(i), m_labeled.get(j));
				m_kUL.add(new _Node(i, j+U, similarity));
			}
			
			for(_Node n:m_kUL) {
				Wij[n.m_i][n.m_j] = n.m_sim;
				Wij[n.m_j][n.m_i] = n.m_sim;
			}
			m_kUL.clear();
		}
		
		/****Construct the C+scale*\Delta matrix and Y vector.****/
		double scale = -m_alpha / (m_k + m_beta*m_kPrime);
		double[] Y = new double[U+L];
		for(int i = 0; i < U+L; i++) {
			Wij[i][i] = -Utils.sumOfArray(Wij[i]);
			Utils.scaleArray(Wij[i], scale);
			
			if (i<U) {
				Wij[i][i] += 1.0;
				Y[i] = m_classifier.predict(m_testSet.get(i)); //Multiple learner.
			} else {
				Wij[i][i] += m_M;
				Y[i] = m_M * m_labeled.get(i-U).getYLabel();
			}
		}
		
		/***Perform matrix inverse.****/
		DenseDoubleMatrix2D mat = new DenseDoubleMatrix2D(Wij);
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		DoubleMatrix2D result = alg.inverse(mat);
		
		/*******Show results*********/
		for(int i = 0; i < U; i++){
			double pred = 0;
			for(int j=0; j<U+L; j++)
				pred += result.getQuick(i, j) * Y[j];
			
			m_TPTable[getLabel(pred)][m_testSet.get(i).getYLabel()] += 1;
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
	}
	
	//get the closest int
	private int getLabel(double pred) {
		for(int i=0; i<m_classNo; i++)
			m_cProbs[i] = -Math.abs(i-pred); //-|c-p(c)|
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	@Override
	public int predict(_Doc doc) {
		return -1; //we don't support this
	}
}
