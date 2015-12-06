package Classifier.supervised;

import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures.annotationType;
import utils.Utils;
import Classifier.supervised.NaiveBayes;

/**
 * @author hongning
 * Naive Bayes with EM training
 */
public class NaiveBayesEM extends NaiveBayes {
	
	//parameters to control EM iterations
	double m_converge = 1e-5;
	int m_maxIter = 50;
	
	public NaiveBayesEM(_Corpus c) {
		super(c);
	}

	public NaiveBayesEM(int classNo, int featureSize) {
		super(classNo, featureSize);
	}

	public NaiveBayesEM(_Corpus c, boolean presence, double deltaY,
			double deltaXY) {
		super(c, presence, deltaY, deltaXY);
	}
	
	public void setEMParam(int maxIter, double converge) {
		m_maxIter = maxIter;
		m_converge = converge;
	}

	@Override
	protected void init() {		
		for(_Doc doc: m_trainSet){
			if (doc.getAnnotationType()==annotationType.UNANNOTATED)
				doc.setTopics(m_classNo, m_deltaY);//create the storage space
		}
		
		MStep(m_trainSet, 0);
	}
	
	double EStep(Collection<_Doc> trainSet) {
		double likelihood = 0;
		for(_Doc doc: m_trainSet){
			score(doc, 0);//to compute p(x|y)p(y) and store it in m_cProbs
			if (doc.getAnnotationType()==annotationType.UNANNOTATED) {//unlabeled data
				double sumY = Utils.logSumOfExponentials(m_cProbs);
				for(int i=0; i<m_classNo; i++) {
					doc.m_sstat[i] = Math.exp(m_cProbs[i] - sumY); // p(y|x)
					likelihood += doc.m_sstat[i] * m_cProbs[i]; // p(x)
				}
			} else if (doc.getAnnotationType()==annotationType.ANNOTATED || doc.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED) {//labeled data
				likelihood += m_cProbs[doc.getYLabel()]; //p(x, y=Y)
			}
		}
		
		return likelihood;
	}
	 
	void MStep(Collection<_Doc> trainSet, int iter) {
		super.init();
		
		for(_Doc doc: trainSet){
			if (doc.getAnnotationType()==annotationType.ANNOTATED || doc.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED) {// labeled data
				int label = doc.getYLabel();
				m_pY[label] ++;
				for(_SparseFeature sf: doc.getSparse())
					m_Pxy[label][sf.getIndex()] += m_presence?1.0:sf.getValue();
			} else if (iter>0 && doc.getAnnotationType()==annotationType.UNANNOTATED) {// unlabeled data
				double[] label = doc.m_sstat;
				for(int i=0; i<m_classNo; i++) {
					m_pY[i] += label[i];
					for(_SparseFeature sf: doc.getSparse())
						m_Pxy[i][sf.getIndex()] += (m_presence?1.0:sf.getValue()) * label[i];
				}
			} 
		}

		
		//normalization
		double sumY = Math.log(Utils.sumOfArray(m_pY) + m_deltaY * m_classNo);
		for(int i = 0; i < m_classNo; i++){
			m_pY[i] = Math.log(m_pY[i] + m_deltaY) - sumY;
			double sumX = Math.log(Utils.sumOfArray(m_Pxy[i]) + m_featureSize*m_deltaXY);
			for(int j = 0; j < m_featureSize; j++)
				m_Pxy[i][j] = Math.log(m_deltaXY+m_Pxy[i][j]) - sumX;
		}
	}
	
	//EM-training on the data set.
	public void train(Collection<_Doc> trainSet){
		init();
		
		double current = 0, last = -1.0, converge = 1.0;
		int iter = 1;
		
		do {
			current = EStep(trainSet);
			MStep(trainSet, iter);
			
			if (iter==1)
				converge = 1.0;
			else
				converge = (last-current)/last;
			
			last = current;
			System.out.format("Iteration number %d, loglikelihood %f converge to %f \n", iter, current, converge);
			
			iter ++;
		} while(iter<m_maxIter && converge>m_converge);
		
		System.out.format("NaiveBayes-EM converge to %.4f after %d iterations...\n", converge, iter);
	}
	
	@Override
	public double test() {// we have override it because we want to test only against original newEgg dataset
		double acc = 0;
		int testSize = 0;
		for(_Doc doc: m_testSet){
			// only Testing against newEggset
			if(doc.getAnnotationType()==annotationType.ANNOTATED || doc.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED){
				testSize++;
				doc.setPredictLabel(predict(doc)); //Set the predict label according to the probability of different classes.
				int pred = doc.getPredictLabel(), ans = doc.getYLabel();
				m_TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.

				if (pred != ans) {
					if (m_debugOutput!=null && Math.random()<0.2)//try to reduce the output size
						debug(doc);
				} else {//also print out some correctly classified samples
					if (m_debugOutput!=null && Math.random()<0.02)
						debug(doc);
					acc ++;
				}
			}
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		return acc /testSize;
	}
}