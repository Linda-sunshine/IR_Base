package Classifier.supervised;

import java.util.Collection;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures.annotationType;
import utils.Utils;

public class EMNaiveBayes extends NaiveBayes {
	
	private double logLike = 0.0;
	private int iter;
	private double m_convergence;
	
	//Constructor.
	public EMNaiveBayes(_Corpus c, int numberOfIteration, double convergence){
		super(c);
		iter = numberOfIteration;
		m_convergence = convergence;
	}
	
	//Constructor.
	public EMNaiveBayes(int classNo, int featureSize,int numberOfIteration, double convergence){
		super(classNo, featureSize);
		iter = numberOfIteration;
		m_convergence = convergence;
	}
	
	//Constructor.
	public EMNaiveBayes(_Corpus c, boolean presence, double deltaY, double deltaXY,int numberOfIteration, double convergence){
		super(c,presence,deltaY,deltaXY);
		iter = numberOfIteration;
		m_convergence = convergence;
	}
	
	@Override
	public String toString() {
		return String.format("EM-Naive Bayes[C:%d, F:%d]", m_classNo, m_featureSize);
	}

	
	protected void clearCache(Collection<_Doc> trainSet) {		
		for(_Doc doc: m_trainSet){
			if (doc.getAnnotationType()==annotationType.UNANNOTATED)
				doc.setTopics(m_classNo, m_deltaY);//create the storage space
		}
	}
	
	protected void calculateLikelihoodForLabelledDocs(Collection<_Doc> trainSet){
		// likelihood calculation
		for(_Doc d: trainSet){
			if(d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED ){// from NewEgg 
				d.m_sstat = new double[m_classNo];
				int i = d.getYLabel();
				d.m_sstat[i] = m_pY[i];
				for(_SparseFeature f:d.getSparse())
					d.m_sstat[i]  += m_Pxy[i][f.getIndex()] * (m_presence?1.0:f.getValue());
				logLike += d.m_sstat[i];
			}
		}
	}
	
	public void initialTrain(Collection<_Doc> trainSet){
		//initial Naive Bayes using only newEgg data
		for(_Doc doc: trainSet){
			if(doc.getAnnotationType()==annotationType.ANNOTATED || doc.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED){// from NewEgg 
				int label = doc.getYLabel();
				m_pY[label] ++;
				for(_SparseFeature sf: doc.getSparse())
					m_Pxy[label][sf.getIndex()] += m_presence?1.0:sf.getValue();
				}
		}
		//normalization
		double sumY = Math.log(Utils.sumOfArray(m_pY) + m_deltaY * m_classNo);
		for(int i = 0; i < m_classNo; i++){
			m_pY[i] = Math.log(m_pY[i] + m_deltaY) -sumY;//up to a constant since normalization of this is not important
			double sum = Math.log(Utils.sumOfArray(m_Pxy[i]) + m_featureSize*m_deltaXY);
			for(int j = 0; j < m_featureSize; j++)
				m_Pxy[i][j] = Math.log(m_deltaXY+m_Pxy[i][j]) - sum;
		}
	}
	
	
	//Train the data set.
	@Override
	public void train(Collection<_Doc> trainSet){
		
		super.init();
		clearCache(trainSet);
		//initial Naive Bayes estimation on labelled dataset
		initialTrain(trainSet);
		
		double delta, current, last=0.0;
		
		//EM algorithm start 
		int i = 0;
		while(i<iter){
			logLike = 0.0;
			calculateLikelihoodForLabelledDocs(trainSet);
			// E-step only use unlabeled dataset
			E_step(trainSet);
			// M-step uses both labeled and unlabeled dataset
			M_step(trainSet);
			
			current = logLike;
			if (i>0)
				delta = (last-current)/last;
			else
				delta = 1.0;
			last = current;
			System.out.format("Iteration number %d, loglikelihood %f converge to %f \n", i, logLike, delta);
			i++;
			
			if (Math.abs(delta)<m_convergence)
				break;//to speed-up, we don't need to compute likelihood in many cases
		}
	}
	
	// Use amazon dataset to estimate the label of the unlabelled doc 
	public void E_step(Collection<_Doc> docSet){
		for(_Doc d: docSet){
			if(d.getAnnotationType()==annotationType.UNANNOTATED){// using the unlabelled Amazon dataset 
				// all computation in logscale
				for(int i = 0; i < m_classNo; i++){
					d.m_sstat[i] = m_pY[i];
					for(_SparseFeature f:d.getSparse())
						d.m_sstat[i]  += m_Pxy[i][f.getIndex()] * (m_presence?1.0:f.getValue());
					}
				//normalize inside document
				double norm = Double.NEGATIVE_INFINITY; // log 0;
				for(int i = 0; i < m_classNo; i++){
					norm = Utils.logSum(norm, d.m_sstat[i]);
				}
				// also accumulate the loglikelihood
				for(int i = 0; i < m_classNo; i++){
					d.m_sstat[i] -= norm;
					logLike += d.m_sstat[i];
				}
			}
		}
		//clearing m_pY and m_PXY for M_step in this iteration
		super.init();
	}

	public void M_step(Collection<_Doc> docSet){
		//Now use both newEgg and amazon
		//here doing all thing in normal scale
		for(_Doc d: docSet){
			if(d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED){
				int label = d.getYLabel();
				m_pY[label] ++;
				for(_SparseFeature sf: d.getSparse())
					m_Pxy[label][sf.getIndex()] += m_presence?1.0:sf.getValue();
			}
			else if(d.getAnnotationType()==annotationType.UNANNOTATED){
				for(int label = 0; label < m_classNo; label++){
					m_pY[label] += Math.exp(d.m_sstat[label]);
					for(_SparseFeature sf: d.getSparse())
						m_Pxy[label][sf.getIndex()] += m_presence?1.0:Math.exp(d.m_sstat[label])*sf.getValue();
				}
			}
		}
		//normalization
		//placing all values in logscale
		double sumY = Math.log(Utils.sumOfArray(m_pY) + m_deltaY * m_classNo);
		for(int i = 0; i < m_classNo; i++){
			m_pY[i] = Math.log(m_pY[i] + m_deltaY) - sumY;//up to a constant since normalization of this is not important
			double sum = Math.log(Utils.sumOfArray(m_Pxy[i]) + m_featureSize*m_deltaXY);
			for(int j = 0; j < m_featureSize; j++)
				m_Pxy[i][j] = Math.log(m_deltaXY+m_Pxy[i][j]) - sum;
		}

	}
	

	/*
	@Override
	public double test() {// we have override it because we want to test only against original newEgg dataset
		double acc = 0;
		int testSize = 0;
		for(_Doc doc: m_testSet){
			// only Testing against newEggset
			if(doc.getSourceType()==2){
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
	}*/
}
