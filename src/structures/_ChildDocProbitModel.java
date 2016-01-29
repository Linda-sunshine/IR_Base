package structures;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import Jama.Matrix;
import cern.jet.random.Normal;
import cern.jet.random.engine.DRand;
import cern.jet.random.engine.RandomEngine;
import utils.Utils;

public class _ChildDocProbitModel extends _ChildDoc{
	public double[] m_xIndicator;
	public double[][] m_probitFeature;
	public HashMap<Integer, double[]> m_fixedMuPartMap;
	public HashMap<Integer, Double> m_fixedSigmaPartMap;
	public Normal m_Normal;
	
	public _ChildDocProbitModel(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		RandomEngine engine = new DRand();
		m_Normal = new Normal(0, 1, engine);
		
		m_fixedMuPartMap = new HashMap<Integer, double[]>();
		m_fixedSigmaPartMap = new HashMap<Integer, Double>();
	}
	
	public void setProbitFeature(int wIndex, double idfCorpus, double tfChild, double tfParent){
		int i = 0;
		
		m_probitFeature[wIndex][i] = idfCorpus;
		i ++;
		
		m_probitFeature[wIndex][i] = tfChild;
		i ++;
		
		m_probitFeature[wIndex][i] = tfParent;
	}
	
	public void setFixedFeatureValueMap(){
		double[][] otherFeatures = new double[m_probitFeature.length-1][m_probitFeature[0].length];
		double[] feature = new double[m_probitFeature[0].length];
	
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
				m_fixedSigmaPartMap.put(wid, sigma);
			}
		}
	}
	
	public void setTopics4Gibbs(int k, double alpha, _Corpus c){
		if(m_topics==null || m_topics.length != k){
			m_topics = new double[k];
			m_sstat =  new double[k];
		}
		
		Arrays.fill(m_sstat, alpha);
		
		int docSize = getTotalDocLength();
		if(m_words == null || m_words.length!=docSize){
			m_words = new int[docSize];
			m_topicAssignment = new int[docSize];
			m_xIndicator = new double[docSize];
		}
		
		//initial probit feature value
		String[] probitFeatureName = new String[]{"IDF copurs", "TF childDoc", "TF parentDoc"};
		int probitFeatureNum = probitFeatureName.length;
		m_probitFeature = new double[m_words.length][probitFeatureNum];
		
		int wIndex = 0;
		if(m_rand == null)
			m_rand = new Random();
		
		int wid, gammaSize = m_xSstat.length;
		double tfChild = 0.0;
		double tfParent = 0.0;
		for(_SparseFeature fv: m_x_sparse){
			wid = fv.getIndex();
			tfChild = fv.getValue();
			
			//get probit feature values
			String featureName = c.m_features.get(wid);
			int[] DFarray = c.m_featureStat.get(featureName).getDF();
			double DF = Utils.sumOfArray(DFarray);
			double idfCorpus = Math.log(1+c.getSize()/DF);
			
			for(_SparseFeature pFv: m_parentDoc.getSparse()){
				if(pFv.getIndex() == wid){
					tfParent = pFv.getValue(); 
				}
			}
			
			for(int j=0; j<fv.getValue(); j++){
				int xIndex = 0;
				m_words[wIndex] = wid;
				
				setProbitFeature(wIndex, idfCorpus, tfChild, tfParent);
				
				m_topicAssignment[wIndex] = m_rand.nextInt(k);
				// (0,1) gaussian initialize
				double gaussianVal = m_Normal.nextDouble();
				m_xIndicator[wIndex] = gaussianVal;
				
				if (m_xIndicator[wIndex] > 0)
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
			
			if(m_xIndicator != null){
				x = m_xIndicator[s];
				m_xIndicator[s] = m_xIndicator[i];
				m_xIndicator[i] = x;
			}
		}
	}

}
