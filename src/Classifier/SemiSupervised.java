package Classifier;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

public class SemiSupervised extends BaseClassifier{
	
	protected double m_alpha; //Weight coefficient between unlabeled node and labeled node.
	protected double m_beta; //Weight coefficient between unlabeled node and unlabeled node.
	protected double m_M; //Influence of labeled node.
	protected int m_k; // k labeled nodes.
	protected int m_kPrime;//k' unlabeled nodes.
	
	private int m_U, m_L;
	private double[] m_cache; // cache the similarity computation results given the similarity metric is symmetric
	
	protected MyPriorityQueue<_RankItem> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
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
	
	@Override
	public String toString() {
		return String.format("Transductive Learning[C:%s, k:%d, k':%d]", m_classifier, m_k, m_kPrime);
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
	
	private void initCache() {
		m_cache = new double[m_U*(2*m_L+m_U-1)/2];//specialized for the current matrix structure
	}
	
	private int encode(int i, int j) {
		if (i>j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return (2*(m_U+m_L-1)-i)/2*(i+1) - ((m_U+m_L)-j);//specialized for the current matrix structure
	}
	
	private void setCache(int i, int j, double v) {
		m_cache[encode(i,j)] = v;
	}
	
	private double getCache(int i, int j) {
		return m_cache[encode(i,j)];
	}
	
	//Test the data set.
	@Override
	public void test(){
		
		double similarity = 0;
		m_L = m_labeled.size();
		m_U = m_testSet.size();
		
		/***Set up cache structure for efficient computation.****/
		initCache();
		
		/***pre-compute the full similarity matrix (except the diagonal).****/
		for(int i = 0; i < m_U; i++){
			for(int j = i+1; j < m_U; j++){//to save computation since our similarity metric is symmetric
				similarity = m_beta * Utils.calculateSimilarity(m_testSet.get(i), m_testSet.get(j));
				setCache(i, j, similarity);
			}	

			for(int j = 0; j < m_L; j++){
				similarity = Utils.calculateSimilarity(m_testSet.get(i), m_labeled.get(j));
				setCache(i, m_U+j, similarity);
			}
		}
		
		SparseDoubleMatrix2D mat = new SparseDoubleMatrix2D(m_U+m_L, m_U+m_L);
		
		/***Set up structure for k nearest neighbors.****/
		m_kUU = new MyPriorityQueue<_RankItem>(m_kPrime);
		m_kUL = new MyPriorityQueue<_RankItem>(m_k);
		
		/****Construct the C+scale*\Delta matrix and Y vector.****/
		double scale = -m_alpha / (m_k + m_beta*m_kPrime), sum, value;
		double[] Y = new double[m_U+m_L];
		for(int i = 0; i < m_U; i++) {
			//set the part of unlabeled nodes. U-U
			for(int j=0; j<m_U; j++) {
				if (j==i)
					continue;
				
				//differentiate reviews from different products
				if (m_testSet.get(i).sameProduct(m_testSet.get(j)))
					m_kUU.add(new _RankItem(j, getCache(i,j)));
				else
					m_kUU.add(new _RankItem(j, getCache(i,j)/2));
			}
			
			sum = 0;
			for(_RankItem n:m_kUU) {
				value = Math.max(n.m_value, mat.getQuick(i, n.m_index)/scale);//recover the original Wij
				mat.setQuick(i, n.m_index, scale * value);
				mat.setQuick(n.m_index, i, scale * value);
				sum += value;
			}
			m_kUU.clear();
			
			//Set the part of labeled and unlabeled nodes. L-U and U-L
			for(int j=0; j<m_L; j++) {
				if (m_testSet.get(i).sameProduct(m_labeled.get(j)))
					m_kUL.add(new _RankItem(m_U+j, getCache(i,m_U+j)));
				else
					m_kUL.add(new _RankItem(m_U+j, getCache(i,m_U+j)/2));
			}
			
			for(_RankItem n:m_kUL) {
				value = Math.max(n.m_value, mat.getQuick(i, n.m_index)/scale);//recover the original Wij
				mat.setQuick(i, n.m_index, scale * value);
				mat.setQuick(n.m_index, i, scale * value);
				sum += value;
			}
			mat.setQuick(i, i, 1-scale*sum);
			m_kUL.clear();
			
			//set up the Y vector for unlabeled data
			Y[i] = m_classifier.predict(m_testSet.get(i)); //Multiple learner.
		}
		
		for(int i=m_U; i<m_L+m_U; i++) {
			sum = 0;
			for(int j=0; j<m_U; j++) 
				sum += mat.getQuick(i, j);
			mat.setQuick(i, i, m_M-sum); // scale has been already applied in each cell
			
			//set up the Y vector for labeled data
			Y[i] = m_M * m_labeled.get(i-m_U).getYLabel();
		}
		
		/***Perform matrix inverse.****/
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		DoubleMatrix2D result = alg.inverse(mat);
		
		/*******Show results*********/
		for(int i = 0; i < m_U; i++){
			double pred = 0;
			for(int j=0; j<m_U+m_L; j++)
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
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation){
		
	}
}
