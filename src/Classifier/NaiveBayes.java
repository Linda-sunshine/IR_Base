package Classifier;

import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

public class NaiveBayes extends BaseClassifier {
	private double[][] m_Pxy; // p(X|Y)
	private double[] m_pY;//p(Y)
	private boolean m_presence;
	private double m_deltaY; // for smoothing p(Y) purpose;
	private double m_deltaXY; // for smoothing p(X|Y) purpose;
	
	//Constructor.
	public NaiveBayes(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		m_Pxy = new double [m_classNo][featureSize];
		m_pY = new double [m_classNo];
		
		m_presence = false;
		m_deltaY = 0.1;
		m_deltaXY = 0.1;
	}
	
	//Constructor.
	public NaiveBayes(_Corpus c, int classNumber, int featureSize, boolean presence, double deltaY, double deltaXY){
		super(c, classNumber, featureSize);
		m_Pxy = new double [m_classNo][featureSize];
		m_pY = new double [m_classNo];
		
		m_presence = presence;
		m_deltaY = deltaY;
		m_deltaXY = deltaXY;
	}
	
	@Override
	public String toString() {
		return String.format("Naive Bayes[C:%d, F:%d]", m_classNo, m_featureSize);
	}
	
	protected void init() {
		for(int i=0; i<m_classNo; i++) {
			Arrays.fill(m_Pxy[i], 0);
			m_pY[i] = 0;
		}
	}
	
	//Train the data set.
	public void train(Collection<_Doc> trainSet){
		init();
		
		for(_Doc doc: trainSet){
			int label = doc.getYLabel();
			m_pY[label] ++;
			for(_SparseFeature sf: doc.getSparse())
				m_Pxy[label][sf.getIndex()] += m_presence?1:sf.getValue();
		}
		
		//normalization
		for(int i = 0; i < m_classNo; i++){
			m_pY[i] = Math.log(m_pY[i] + m_deltaY);//up to a constant since normalization of this is not important
			double sum = Math.log(Utils.sumOfArray(m_Pxy[i]) + m_featureSize*m_deltaXY);
			for(int j = 0; j < m_featureSize; j++)
				m_Pxy[i][j] = Math.log(m_deltaXY+m_Pxy[i][j]) - sum;
		}
		
		printTopFeatures(5);
	}
		
	//Predict the label for one document.
	@Override
	public int predict(_Doc d){
		for(int i = 0; i < m_classNo; i++){
			m_cProbs[i] = m_pY[i];
			for(_SparseFeature f:d.getSparse())
				m_cProbs[i] += m_Pxy[i][f.getIndex()] * (m_presence?1.0:f.getValue());
		}
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation) {
		
	}
	
	public void printTopFeatures(int topK) {
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(topK, false);
		for(int n=0; n<m_featureSize; n++) {
			for(int i=0; i<m_classNo; i++)
				m_cProbs[i] = m_Pxy[i][n];
			queue.add(new _RankItem(n, Utils.entropy(m_cProbs, true)));
		}
		
		System.out.print("Most discriminative features: ");
		for(_RankItem item:queue) {
			for(int i=0; i<m_classNo; i++)
				m_cProbs[i] = m_Pxy[i][item.m_index];
			System.out.format("%s(%d) ", m_corpus.getFeature(item.m_index), Utils.maxOfArrayIndex(m_cProbs));
		}
		System.out.println();
	}
}
