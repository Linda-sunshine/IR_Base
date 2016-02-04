package structures;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import Analyzer.ParentChildAnalyzer;
import Jama.Matrix;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DRand;
import utils.Utils;

public class _ChildDoc4ProbitModel extends _ChildDoc{
	public double[] m_xIndicatorValue;
	public double[][] m_probitFeature;
	public HashMap<Integer, double[]> m_fixedMuPartMap;
	public HashMap<Integer, Double> m_fixedSigmaPartMap;
	public Normal m_Normal;
	
	public _ChildDoc4ProbitModel(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		m_Normal = new Normal(0, 1, new DRand());
		
		m_fixedMuPartMap = new HashMap<Integer, double[]>();//from word index to a vector
		m_fixedSigmaPartMap = new HashMap<Integer, Double>();
	}
	
	public void setFixedFeatureValueMap(){
		double[][] otherFeatures = new double[m_probitFeature.length-1][ParentChildAnalyzer.ChildDocFeatureSize];
		double[] feature = new double[ParentChildAnalyzer.ChildDocFeatureSize];
	
		for(int n=0; n<m_words.length; n++){
			int wid = m_words[n];
			if (!m_fixedMuPartMap.containsKey(wid)) {
				for (int m = 0; m < m_words.length; m++) {
					for (int f = 0; f < m_probitFeature[0].length; f++) {
						if (m == n) {
							feature[f] = m_probitFeature[m][f];
						} else if (m < n) {
							otherFeatures[m][f] = m_probitFeature[m][f];
						} else {
							otherFeatures[m - 1][f] = m_probitFeature[m][f];
						}
					}
				}

				Matrix featureMatrix = new Matrix(feature, feature.length);

				Matrix otherFeaturesMatrix = new Matrix(otherFeatures);
				Matrix aMatrix = otherFeaturesMatrix.transpose().times(otherFeaturesMatrix);
				Matrix identityMatrix = Matrix.identity(aMatrix.getRowDimension(), aMatrix.getRowDimension());
				aMatrix = aMatrix.plus(identityMatrix.times(0.01));
				Matrix aMatrixInverse = aMatrix.inverse();

				Matrix mm = featureMatrix.transpose().times(aMatrixInverse);
				mm = mm.times(otherFeaturesMatrix.transpose());

				m_fixedMuPartMap.put(wid, mm.getArray()[0]);
				
				double sigma = featureMatrix.transpose().times(aMatrixInverse).times(featureMatrix).get(0, 0)+1;
				m_fixedSigmaPartMap.put(wid, Math.sqrt(sigma));
			}
		}
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha){
		if(m_topics==null || m_topics.length != k){
			m_topics = new double[k];
			m_sstat =  new double[k];
		}
		
		Arrays.fill(m_sstat, alpha);
		
		int docSize = getTotalDocLength();
		if(m_words == null || m_words.length!=docSize){
			m_words = new int[docSize];
			m_topicAssignment = new int[docSize];
			m_xIndicatorValue = new double[docSize];
		}
		
		//initial probit feature value		
		m_probitFeature = new double[m_words.length][];
		
		int wIndex = 0;
		if(m_rand == null)
			m_rand = new Random();
		
		int wid, gammaSize = m_xSstat.length;
		double tfChild = 0.0;
		double tfParent = 0.0;
		for(_SparseFeature fv: m_x_sparse){
			wid = fv.getIndex();
			
			for(int j=0; j<fv.getValue(); j++){
				int xIndex = 0;
				m_words[wIndex] = wid;
				
				m_probitFeature[wIndex] = fv.getValues();
				
				m_topicAssignment[wIndex] = m_rand.nextInt(k);
				// N(0,1) gaussian initialize
				m_xIndicatorValue[wIndex] = m_Normal.nextDouble();
				
				if (m_xIndicatorValue[wIndex] > 0)
					xIndex = 1;
				else
					xIndex = 0;
				m_xTopicSstat[xIndex][m_topicAssignment[wIndex]] ++;
				m_xSstat[xIndex] ++;
				
				wIndex ++;
			}
		}
		
		setFixedFeatureValueMap();
	}

	@Override
	public void permutation(){
		int s, t;
		double x;
		for(int i=m_words.length-1; i>1; i--){
			s = m_rand.nextInt(i);
			
			t = m_words[s];
			m_words[s] = m_words[i];
			m_words[i] = t;
			
			t = m_topicAssignment[s];
			m_topicAssignment[s] = m_topicAssignment[i];
			m_topicAssignment[i] = t;
			
			if(m_xIndicatorValue != null){
				x = m_xIndicatorValue[s];
				m_xIndicatorValue[s] = m_xIndicatorValue[i];
				m_xIndicatorValue[i] = x;
			}
		}
	}

}
